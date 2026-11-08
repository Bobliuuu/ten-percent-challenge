"""X03 shipped-format library (this file ships inside every pack: its bytes are counted).
Contents: Rot (sign+QR+Hadamard incoherence rotation), mixed-radix index packing (groups of 8 indices, exact ceil(8 log2 K) bits per group),
lane-interleaved rANS for log-delta coded scales, tiered embedding/head quantiser, and the pack-level reader used by decode_pack.py."""
import math, zlib, json
import numpy as np, torch

# ------------------------------------------------------------------ rotation
def had(p):
    Hm = torch.ones(1, 1)
    while Hm.shape[0] < p: Hm = torch.cat([torch.cat([Hm, Hm], 1), torch.cat([Hm, -Hm], 1)], 0)
    return Hm / math.sqrt(p)
class Rot:
    def __init__(s, n, seed, dev="cpu"):
        p = n & (-n); m = n // p; g = torch.Generator().manual_seed(seed)
        s.sg = (torch.randint(0, 2, (n,), generator=g) * 2 - 1).float().to(dev)
        Q = torch.linalg.qr(torch.randn(m, m, generator=g, dtype=torch.float64))[0] if m > 1 else torch.ones(1, 1, dtype=torch.float64)
        s.Q = Q.float().to(dev); s.H = had(p).to(dev); s.m = m; s.p = p
    def fwd(s, X):
        sh = X.shape; Y = (X * s.sg).reshape(-1, s.m, s.p); Y = torch.matmul(s.Q, Y); Y = torch.matmul(Y, s.H); return Y.reshape(sh)
    def inv(s, X):
        sh = X.shape; Y = X.reshape(-1, s.m, s.p); Y = torch.matmul(s.Q.T, Y); Y = torch.matmul(Y, s.H); return Y.reshape(sh) * s.sg

# ------------------------------------------------------------------ mixed radix packing (8 indices per group, 4 x 32-bit limbs)
G = 8; MASK32 = 0xFFFFFFFF
def group_bits(K): return (K ** G - 1).bit_length()
def packed_nbytes(n, K): return (((n + G - 1) // G) * group_bits(K) + 7) // 8
def pack_idx(idx, K):
    """idx: 1-D integer tensor (values < K) -> uint8 numpy bytes"""
    dev = idx.device; n = idx.numel(); pad = (-n) % G
    a = torch.cat([idx.reshape(-1).to(torch.int64), torch.zeros(pad, dtype=torch.int64, device=dev)]).reshape(-1, G); N = a.shape[0]; B = group_bits(K)
    v = [torch.zeros(N, dtype=torch.int64, device=dev) for _ in range(4)]
    for j in range(G):
        carry = a[:, j].clone()
        for i in range(4):
            t = v[i] * K + carry; v[i] = t & MASK32; carry = t >> 32
    assert int(carry.max()) == 0
    out = []
    CH = 1 << 20
    # bit matrix (N, B) little-endian within the group, groups concatenated
    bits = torch.empty(N, B, dtype=torch.uint8, device=dev)
    for b in range(B): bits[:, b] = ((v[b // 32] >> (b % 32)) & 1).to(torch.uint8)
    flat = bits.reshape(-1); nb = flat.numel(); flat = torch.cat([flat, torch.zeros((-nb) % 8, dtype=torch.uint8, device=dev)]).reshape(-1, 8)
    w = (2 ** torch.arange(8, dtype=torch.int64, device=dev))
    return (flat.to(torch.int64) * w).sum(1).to(torch.uint8).cpu().numpy()
def unpack_idx(buf, n, K, dev="cpu"):
    N = (n + G - 1) // G; B = group_bits(K)
    by = torch.from_numpy(np.frombuffer(buf, dtype=np.uint8).copy()).to(dev).to(torch.int64)
    bits = ((by[:, None] >> torch.arange(8, device=dev)) & 1).reshape(-1)[:N * B].reshape(N, B)
    v = [torch.zeros(N, dtype=torch.int64, device=dev) for _ in range(4)]
    for b in range(B): v[b // 32] |= bits[:, b] << (b % 32)
    out = torch.empty(N, G, dtype=torch.int64, device=dev)
    for j in reversed(range(G)):
        rem = torch.zeros(N, dtype=torch.int64, device=dev)
        for i in reversed(range(4)):
            t = (rem << 32) | v[i]; v[i] = t // K; rem = t % K
        out[:, j] = rem
    return out.reshape(-1)[:n]

# ------------------------------------------------------------------ rANS (lane-interleaved, 32-bit state, 16-bit renorm), static model
PB = 14; LANES = 1024
def build_model(sym, A):
    cnt = np.bincount(sym, minlength=A).astype(np.float64); f = np.maximum(1, np.floor(cnt / cnt.sum() * (1 << PB))).astype(np.int64)
    f[cnt == 0] = 0
    while f.sum() > (1 << PB): f[np.argmax(f)] -= 1
    while f.sum() < (1 << PB): f[np.argmax(f)] += 1
    return f
def rans_encode(sym, f):
    """sym: 1-D int array -> (bytes, n). Symbols laid out lane-major (lane = i % LANES)."""
    n = len(sym); T = (n + LANES - 1) // LANES; pad = T * LANES - n; s = np.concatenate([sym, np.full(pad, int(np.argmax(f)), sym.dtype)]).reshape(T, LANES)
    c = np.concatenate([[0], np.cumsum(f)[:-1]]).astype(np.uint64); fu = f.astype(np.uint64)
    x = np.full(LANES, 1 << 16, dtype=np.uint64); words = []
    for t in range(T - 1, -1, -1):
        st = s[t]; fs = fu[st]; xmax = ((np.uint64(1 << 16) >> np.uint64(PB)) << np.uint64(16)) * fs
        m = x >= xmax
        if m.any(): words.append((x[m] & np.uint64(0xFFFF)).astype(np.uint16)); x = np.where(m, x >> np.uint64(16), x)
        x = ((x // fs) << np.uint64(PB)) + (x % fs) + c[st]
    w = np.concatenate(words) if words else np.zeros(0, np.uint16)
    return x.astype(np.uint32).tobytes() + w.tobytes()
def rans_decode(buf, n, f):
    T = (n + LANES - 1) // LANES; x = np.frombuffer(buf[:4 * LANES], dtype=np.uint32).astype(np.uint64).copy(); words = np.frombuffer(buf[4 * LANES:], dtype=np.uint16)[::-1].astype(np.uint64)
    c = np.concatenate([[0], np.cumsum(f)[:-1]]).astype(np.uint64); fu = f.astype(np.uint64)
    tab = np.zeros(1 << PB, dtype=np.int64)
    for a in range(len(f)): tab[int(c[a]):int(c[a]) + int(f[a])] = a
    out = np.empty((T, LANES), dtype=np.int64); pos = 0; MASK = np.uint64((1 << PB) - 1)
    for t in range(T):
        sl = (x & MASK).astype(np.int64); sy = tab[sl]; out[t] = sy
        x = fu[sy] * (x >> np.uint64(PB)) + (x & MASK) - c[sy]
        need = x < np.uint64(1 << 16)
        if need.any():
            lanes = np.nonzero(need)[0][::-1]; k = len(lanes); x[lanes] = (x[lanes] << np.uint64(16)) | words[pos:pos + k]; pos += k
    return out.reshape(-1)[:n]

# ------------------------------------------------------------------ embedding / head quantiser
GS = 128
LEVELS = {1: [-0.7979, 0.7979], 2: [-1.5104, -0.4528, 0.4528, 1.5104], 3: [-2.1520, -1.3440, -0.7560, -0.2451, 0.2451, 0.7560, 1.3440, 2.1520]}
LS_STEP = 32     # scale code: ls = round(log2(scale) * 32); symbol = clip(ls - pred, -128, 127) + 128; pred = rounded running mean of the row's previous ls (first group: row-constant)
def ls_symbols(ls, base=0):
    """ls: (r, ng) int64 (numpy) -> symbols int64 (r, ng) via causal running-mean prediction (feedback uses the decoder-side reconstruction)"""
    r, ng = ls.shape; sym = np.empty_like(ls); csum = np.zeros(r, np.int64)
    for j in range(ng):
        pred = np.full(r, base, np.int64) if j == 0 else np.floor_divide(2 * csum + j, 2 * j)    # rounded mean of previous j reconstructed values
        sym[:, j] = np.clip(ls[:, j] - pred, -128, 127) + 128; csum += sym[:, j] - 128 + pred
    return sym
def ls_from_symbols(sym, base=0):
    r, ng = sym.shape; ls = np.empty_like(sym); csum = np.zeros(r, np.int64)
    for j in range(ng):
        pred = np.full(r, base, np.int64) if j == 0 else np.floor_divide(2 * csum + j, 2 * j)
        ls[:, j] = sym[:, j] - 128 + pred; csum += ls[:, j]
    return ls
def clamp_ls(ls, base=0):
    """apply the same clamp as the symbol coding so that encoder-side quantisation uses decoder-exact scales"""
    return ls_from_symbols(ls_symbols(ls, base), base)
def quant_embhead(w, tag, tiers, coded, dev="cuda", chunk=4096):
    """w: (V, hd) tensor. tiers: np int array (V,) of bits in {1,2,3} (A0: all 2). Returns dict with payload bytes per tier (exact accounting)."""
    V, hd = w.shape; rot = Rot(hd, zlib.crc32(tag.encode()), dev); out = dict(tag=tag, V=V, hd=hd, coded=coded, tiers=tiers, parts={}); ngr = hd // GS
    def rotated(rr):
        x = rot.fwd(w[torch.from_numpy(rr)].float().to(dev)); xg = x.reshape(len(rr), ngr, GS); return xg, xg.pow(2).mean(2).sqrt().clamp(min=1e-8)
    for bits in sorted(set(tiers.tolist())):
        rows = np.nonzero(tiers == bits)[0]; part = dict(bits=bits, n_rows=len(rows), rows=rows); L = torch.tensor(LEVELS[bits], device=dev)
        if coded:
            raw = np.concatenate([torch.round(torch.log2(rotated(rows[a:a + chunk])[1]) * LS_STEP).to(torch.int64).cpu().numpy() for a in range(0, len(rows), chunk)])
            base = int(np.median(raw[:, 0])); lsc = clamp_ls(raw, base); sym = ls_symbols(lsc, base); f = build_model(sym.reshape(-1), 256)
            part.update(ls_base=base, ls_model=f.astype(np.uint16).tobytes(), ls_rans=rans_encode(sym.reshape(-1), f), ls_shape=raw.shape)
        idxs = []; scs = []
        for a in range(0, len(rows), chunk):
            xg, sc = rotated(rows[a:a + chunk])
            scq = torch.exp2(torch.from_numpy(lsc[a:a + chunk]).float().to(dev) / LS_STEP) if coded else sc.half().float()
            idx = ((xg / scq.unsqueeze(2)).unsqueeze(-1) - L).abs().argmin(-1); idxs.append(pack_idx(idx.reshape(-1), 1 << bits))
            if not coded: scs.append(scq.half().cpu().numpy())
        part["idx"] = b"".join(i.tobytes() if hasattr(i, "tobytes") else i for i in idxs)
        if not coded: part["scales"] = np.concatenate(scs).tobytes()
        out["parts"][bits] = part
    return out
def embhead_nbytes(e):
    n = 0
    for b, p in e["parts"].items():
        n += len(p["idx"]) + (len(p["ls_model"]) + len(p["ls_rans"]) if e["coded"] else len(p["scales"]))
    return n + (e["V"] + 3) // 4       # tier map 2 bits per row (counted even for the single-tier case)
def dequant_embhead(e, dev="cpu", chunk=4096):
    V, hd = e["V"], e["hd"]; rot = Rot(hd, zlib.crc32(e["tag"].encode()), dev); W = torch.empty(V, hd, dtype=torch.bfloat16); ngr = hd // GS
    for bits, p in e["parts"].items():
        rows = p["rows"]; idx_all = unpack_idx(p["idx"], len(rows) * hd, 1 << bits, dev).reshape(len(rows), hd)
        if e["coded"]:
            f = np.frombuffer(p["ls_model"], dtype=np.uint16).astype(np.int64); sym = rans_decode(p["ls_rans"], int(np.prod(p["ls_shape"])), f).reshape(p["ls_shape"])
            sc_all = torch.exp2(torch.from_numpy(ls_from_symbols(sym, p['ls_base'])).float() / LS_STEP)
        else: sc_all = torch.from_numpy(np.frombuffer(p["scales"], dtype=np.float16).copy().reshape(len(rows), ngr)).float()
        L = torch.tensor(LEVELS[bits], device=dev)
        for a in range(0, len(rows), chunk):
            ii = idx_all[a:a + chunk]; sc = sc_all[a:a + chunk].to(dev); q = (L[ii].reshape(len(ii), ngr, GS) * sc.unsqueeze(2)).reshape(len(ii), hd)
            W[torch.from_numpy(rows[a:a + chunk])] = rot.inv(q).to(torch.bfloat16).cpu()
    return W
def save_embhead(e, path_prefix):
    """-> list of files written; layout: <prefix>.json (header) + <prefix>.bin"""
    hdr = dict(tag=e["tag"], V=e["V"], hd=e["hd"], coded=e["coded"], parts=[]); blob = bytearray()
    tier_map = np.zeros(e["V"], np.uint8)
    for b, p in e["parts"].items(): tier_map[p["rows"]] = b
    tm = np.zeros(((e["V"] + 3) // 4) * 4, np.uint8); tm[:e["V"]] = tier_map; tm = tm.reshape(-1, 4); tmb = (tm[:, 0] | (tm[:, 1] << 2) | (tm[:, 2] << 4) | (tm[:, 3] << 6)).astype(np.uint8).tobytes()   # values 1..3 fit in 2 bits
    blob += tmb
    def put(bs):
        o = len(blob); blob.extend(bs); return [o, len(bs)]
    for b, p in e["parts"].items():
        d = dict(bits=b, n_rows=p["n_rows"], idx=put(p["idx"]))
        if e["coded"]: d.update(ls_base=p["ls_base"], ls_model=put(p["ls_model"]), ls_rans=put(p["ls_rans"]), ls_shape=list(p["ls_shape"]))
        else: d.update(scales=put(p["scales"]))
        hdr["parts"].append(d)
    open(path_prefix + ".bin", "wb").write(bytes(blob)); open(path_prefix + ".json", "w").write(json.dumps(hdr, separators=(",", ":")))
def load_embhead(path_prefix):
    hdr = json.load(open(path_prefix + ".json")); blob = open(path_prefix + ".bin", "rb").read(); V = hdr["V"]
    tmb = np.frombuffer(blob[:(V + 3) // 4], dtype=np.uint8); tm = np.stack([(tmb >> s) & 3 for s in (0, 2, 4, 6)], 1).reshape(-1)[:V]
    e = dict(tag=hdr["tag"], V=V, hd=hdr["hd"], coded=hdr["coded"], parts={})
    get = lambda o, ln: blob[o:o + ln]
    for d in hdr["parts"]:
        p = dict(bits=d["bits"], n_rows=d["n_rows"], rows=np.nonzero(tm == d["bits"])[0], idx=get(*d["idx"]))
        if hdr["coded"]: p.update(ls_base=d["ls_base"], ls_model=get(*d["ls_model"]), ls_rans=get(*d["ls_rans"]), ls_shape=tuple(d["ls_shape"]))
        else: p.update(scales=get(*d["scales"]))
        e["parts"][d["bits"]] = p
    return e

# ------------------------------------------------------------------ body tensor decode (shared by quantiser and decoder)
def decode_tensor(idx, C, sigma, rin, rout):
    Wq = (C[idx].reshape(idx.shape[0], -1)) * sigma; Wq = rout.inv(Wq.T.contiguous()).T.contiguous(); return rin.inv(Wq)
