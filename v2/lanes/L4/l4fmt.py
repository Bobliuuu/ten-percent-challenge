"""L4 packed-body format (L4P1) + standalone loader. Shipped code (counted in bytes).
File = b'L4P1' + blobs + zlib(json header) + u64 header length. Tensors: raw bf16, or group-quantised (codes bit-packed, fp16 scales, zeros bit-packed).
Expert weights are quantised in a seeded block-Hadamard rotated domain (rotation regenerated from the seed, not stored)."""
import json, math, struct, zlib, hashlib, types, os
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
NL, NE, TOPK, HID, FF = 48, 128, 8, 2048, 768
CH = 1 << 25

def pack_bits(c, bits):
    c = c.reshape(-1); k = 8 // math.gcd(8, bits); nb = bits * k // 8; out = []
    for s in range(0, c.numel(), CH):
        x = c[s:s + CH]; pad = (-x.numel()) % k
        if pad: x = torch.cat([x, x.new_zeros(pad)])
        x = x.view(-1, k).long(); v = torch.zeros(x.shape[0], dtype=torch.int64, device=x.device)
        for i in range(k): v |= x[:, i] << (bits * i)
        out.append(torch.stack([(v >> (8 * j)) & 255 for j in range(nb)], 1).to(torch.uint8).reshape(-1))
    return torch.cat(out)

def unpack_bits(b, bits, n):
    k = 8 // math.gcd(8, bits); nb = bits * k // 8; m = (1 << bits) - 1; out = []
    b = b.reshape(-1, nb)
    for s in range(0, b.shape[0], CH // 8):
        x = b[s:s + CH // 8].long(); v = torch.zeros(x.shape[0], dtype=torch.int64, device=x.device)
        for j in range(nb): v |= x[:, j] << (8 * j)
        out.append(torch.stack([(v >> (bits * i)) & m for i in range(k)], 1).to(torch.uint8).reshape(-1))
    return torch.cat(out)[:n]

def had(n):
    H = torch.ones(1, 1)
    while H.shape[0] < n: H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    return H / math.sqrt(n)
def rot_matrix(dim, blk, seed, dev):
    g = torch.Generator().manual_seed(seed); sg = (torch.randint(0, 2, (dim,), generator=g) * 2 - 1).float()
    B = had(blk); Q = torch.zeros(dim, dim)
    for s0 in range(0, dim, blk): Q[s0:s0 + blk, s0:s0 + blk] = B
    return (sg[:, None] * Q).to(dev)

class Writer:
    def __init__(s, path): s.f = open(path, "wb"); s.f.write(b"L4P1"); s.ent = {}
    def put(s, name, data, meta):
        if isinstance(data, torch.Tensor): data = data.cpu().numpy()
        data = np.ascontiguousarray(data).view(np.uint8); off = s.f.tell(); s.f.write(data.tobytes()); s.ent[name] = [off, int(data.size)] + meta
    def raw(s, name, t):   # bf16/fp16/fp32 tensor stored as is
        dt = {torch.bfloat16: "bf16", torch.float16: "f16", torch.float32: "f32"}[t.dtype]
        s.put(name, t.contiguous().view(torch.int16 if t.dtype != torch.float32 else torch.int32).cpu().numpy(), ["r", dt, list(t.shape)])
    def u8(s, name, t): s.put(name, t.to(torch.uint8).cpu().numpy(), ["r", "u8", list(t.shape)])
    def quant(s, name, codes, sc, z, bits):
        s.put(name + ".c", pack_bits(codes, bits), ["c", bits, list(codes.shape)])
        s.put(name + ".s", sc.half().contiguous().view(torch.int16).cpu().numpy(), ["r", "f16", list(sc.shape)])
        s.put(name + ".z", pack_bits(z, bits), ["c", bits, list(z.shape)])
    def blob(s, name, b): s.put(name, np.frombuffer(zlib.compress(b, 9), dtype=np.uint8), ["b"])
    def close(s):
        h = zlib.compress(json.dumps(s.ent, separators=(",", ":")).encode(), 9); s.f.write(h); s.f.write(struct.pack("<Q", len(h))); s.f.close()

class Reader:
    def __init__(s, path):
        s.path = path; sz = os.path.getsize(path)
        with open(path, "rb") as f:
            assert f.read(4) == b"L4P1"; f.seek(sz - 8); n = struct.unpack("<Q", f.read(8))[0]; f.seek(sz - 8 - n); s.ent = json.loads(zlib.decompress(f.read(n)))
    def bytes(s, name):
        e = s.ent[name]
        with open(s.path, "rb") as f: f.seek(e[0]); return f.read(e[1])
    def get(s, name, dev="cuda"):
        e = s.ent[name]; b = np.frombuffer(s.bytes(name), dtype=np.uint8).copy(); k = e[2]
        if k == "b": return zlib.decompress(b.tobytes())
        t = torch.from_numpy(b).to(dev)
        if k == "c": return unpack_bits(t, e[3], int(np.prod(e[4]))).view(*e[4])
        dt, shp = e[3], e[4]
        if dt == "u8": return t.view(*shp)
        if dt == "f32": return t.view(torch.float32).view(*shp)
        return t.view(torch.int16).view({"bf16": torch.bfloat16, "f16": torch.float16}[dt]).view(*shp)
    def deq(s, name, g=128, dev="cuda"):   # float32 dequantised, groups along last dim
        c = s.get(name + ".c", dev).float(); sc = s.get(name + ".s", dev).float(); z = s.get(name + ".z", dev).float()
        sh = c.shape; cg = c.view(*sh[:-1], sh[-1] // g, g)
        return ((cg - z.unsqueeze(-1)) * sc.unsqueeze(-1)).view(sh)
    def overwrite_raw(s, name, t):
        e = s.ent[name]; b = t.contiguous().view(torch.int16).cpu().numpy().tobytes(); assert len(b) == e[1]
        with open(s.path, "r+b") as f: f.seek(e[0]); f.write(b)

def moe_fwd(self, hs):
    B, T, D = hs.shape; x = hs.view(-1, D); lg = self.gate(x)
    p = F.softmax(lg.float(), -1); w, ii = torch.topk(p, TOPK, -1); w = (w / w.sum(-1, keepdim=True)).to(x.dtype)
    out = torch.zeros_like(x); em = F.one_hot(ii, NE).permute(2, 1, 0); hit = (em.sum((1, 2)) > 0).nonzero().flatten().tolist()
    for e in hit:
        k_, t_ = torch.where(em[e]); xe = x[t_]; ex = self.experts[e]
        a = ex.act_fn(ex.gate_proj(xe)) * ex.up_proj(xe); y = ex.down_proj(a); out.index_add_(0, t_, y * w[t_, k_][:, None])
    return out.view(B, T, D)

class GateW(nn.Module):
    def __init__(s, gate): super().__init__(); s.g = gate; s.mask = None
    def forward(s, x):
        lg = s.g(x)
        return lg + s.mask if s.mask is not None else lg

def names(l):
    return [f"L{l}.{n}" for n in ("q", "k", "v", "o")]

def load_packed(path, dev="cuda", hashes=None, log=print):
    """Build the model from the packed body alone (config + tokenizer files inside the file). Returns (lm, tok, Reader). hashes: dict name->sha256 filled with per-tensor sha256 of the bf16 weights."""
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    import tempfile
    rd = Reader(path); td = tempfile.mkdtemp()
    for fn in json.loads(rd.get("files.list")): open(f"{td}/{fn}", "wb").write(rd.get("file." + fn))
    cfg = AutoConfig.from_pretrained(td); tok = AutoTokenizer.from_pretrained(td); tok.padding_side = "left"
    try:
        from transformers.modeling_utils import no_init_weights
        ctx = no_init_weights()
    except Exception:
        import contextlib; ctx = contextlib.nullcontext()
    torch.set_default_dtype(torch.bfloat16)
    with torch.device(dev), ctx: lm = AutoModelForCausalLM.from_config(cfg, torch_dtype=torch.bfloat16)
    torch.set_default_dtype(torch.float32)
    lm.eval()
    for p in lm.parameters(): p.requires_grad_(False)
    hs = (lambda n, t: hashes.__setitem__(n, hashlib.sha256(t.contiguous().view(torch.int16).cpu().numpy().tobytes()).hexdigest())) if hashes is not None else (lambda n, t: None)
    cp = lambda dst, src, n: (dst.data.copy_(src.to(torch.bfloat16)), hs(n, dst.data))
    m = lm.model
    cp(m.embed_tokens.weight, rd.deq("embed", dev=dev), "embed"); cp(lm.lm_head.weight, rd.deq("head", dev=dev), "head"); cp(m.norm.weight, rd.get("norm", dev), "norm")
    kept_all = json.loads(rd.get("kept"))
    for l in range(NL):
        lay = m.layers[l]; ex = lay.mlp.experts; kept = kept_all[l]
        for nm, mod in (("q", "q_proj"), ("k", "k_proj"), ("v", "v_proj"), ("o", "o_proj")):
            w = rd.get(f"L{l}.{nm}.raw", dev) if f"L{l}.{nm}.raw" in rd.ent else rd.deq(f"L{l}.{nm}", dev=dev)
            cp(getattr(lay.self_attn, mod).weight, w, f"L{l}.{nm}")
        for nm, p_ in (("ln1", lay.input_layernorm.weight), ("ln2", lay.post_attention_layernorm.weight), ("qn", lay.self_attn.q_norm.weight), ("kn", lay.self_attn.k_norm.weight)):
            cp(p_, rd.get(f"L{l}.{nm}", dev), f"L{l}.{nm}")
        gw = rd.get(f"L{l}.router", dev); lay.mlp.gate.weight.data.zero_(); lay.mlp.gate.weight.data[kept] = gw; hs(f"L{l}.router", lay.mlp.gate.weight.data)
        rotf = rd.get(f"L{l}.rot", dev).tolist(); Qi = rot_matrix(HID, 256, 1000 + l, dev); Qd = rot_matrix(FF, 256, 2000 + l, dev)
        G = rd.deq(f"L{l}.gate", dev=dev); U = rd.deq(f"L{l}.up", dev=dev); D = rd.deq(f"L{l}.down", dev=dev)
        for j, e in enumerate(kept):
            if rotf[j]:
                q = torch.cat([G[j], U[j]], 0) @ Qi.T; qg, qu = q[:FF], q[FF:]; qd = D[j] @ Qd.T
            else: qg, qu, qd = G[j], U[j], D[j]
            cp(ex[e].gate_proj.weight, qg, f"L{l}.e{e}.g"); cp(ex[e].up_proj.weight, qu, f"L{l}.e{e}.u"); cp(ex[e].down_proj.weight, qd, f"L{l}.e{e}.d")
        for e in range(NE):
            if e not in set(kept): ex[e] = nn.Identity()
        del G, U, D; mk = torch.zeros(NE, device=dev); mk[[e for e in range(NE) if e not in set(kept)]] = float("-inf")
        gw_ = GateW(lay.mlp.gate); gw_.mask = mk; lay.mlp.gate = gw_; lay.mlp.forward = types.MethodType(moe_fwd, lay.mlp)
        torch.cuda.empty_cache()
        if l % 8 == 0: log("loaded layer", l)
    return lm, tok, rd
