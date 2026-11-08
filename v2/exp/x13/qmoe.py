"""X13 runtime + quantiser for Qwen3-30B-A3B-style MoE (pure torch, no transformers MoE code).
Shipped-decoder part: bit unpack/dequant, Model (pruned experts, kept-only router), greedy generate.
Quantiser part: group quant (asymmetric, clip-grid), batched GPTQ, bit packing.  No weight training: closed-form calibration math only."""
import json, os, math, glob, torch, torch.nn.functional as F
G = 128
GRID = (1.0, 0.95, 0.9, 0.85, 0.8, 0.75)

# ---------------- bit packing ----------------
def pack_bits(v, bits):
    """v: integer tensor (any shape) with values < 2**bits -> flat uint8"""
    v = v.reshape(-1).to(torch.int64)
    b = ((v[:, None] >> torch.arange(bits, device=v.device)) & 1).reshape(-1)
    pad = (-b.numel()) % 8
    if pad: b = torch.cat([b, b.new_zeros(pad)])
    return (b.reshape(-1, 8) << torch.arange(8, device=v.device)).sum(-1).to(torch.uint8)

def unpack_bits(p, bits, n):
    b = ((p.to(torch.int64)[:, None] >> torch.arange(8, device=p.device)) & 1).reshape(-1)[: n * bits].reshape(n, bits)
    return (b << torch.arange(bits, device=p.device)).sum(-1)

# ---------------- group quantisation ----------------
def qparams(Wg, bits):
    """Wg [..., g] float32 -> scale [...,1] (fp16-rounded), zp [...,1] (integer-valued float). clip grid search on MSE."""
    qmax = 2 ** bits - 1
    mn = Wg.min(-1, keepdim=True).values; mx = Wg.max(-1, keepdim=True).values
    mid = (mn + mx) / 2; half = (mx - mn) / 2
    best = None
    for s in GRID:
        lo = mid - half * s; hi = mid + half * s
        sc = ((hi - lo) / qmax).clamp_min(1e-5).half().float()
        zp = torch.round(-lo / sc).clamp(0, qmax)
        q = torch.clamp(torch.round(Wg / sc) + zp, 0, qmax)
        err = (((q - zp) * sc - Wg) ** 2).sum(-1, keepdim=True)
        if best is None: best = [err, sc, zp]
        else:
            m = err < best[0]
            best = [torch.where(m, err, best[0]), torch.where(m, sc, best[1]), torch.where(m, zp, best[2])]
    return best[1], best[2]

def rtn(W, bits):
    """W [..., C] -> codes uint8 [...,C], scales fp16 [..., C/G], zp uint8 [..., C/G]"""
    shp = W.shape; Wg = W.float().reshape(*shp[:-1], shp[-1] // G, G)
    sc, zp = qparams(Wg, bits); qmax = 2 ** bits - 1
    q = torch.clamp(torch.round(Wg / sc) + zp, 0, qmax)
    return q.reshape(shp).to(torch.uint8), sc.squeeze(-1).half(), zp.squeeze(-1).to(torch.uint8)

def dequant(codes, sc, zp, dtype=torch.bfloat16):
    """codes [...,C] ints, sc/zp [..., C/G]"""
    shp = codes.shape; q = codes.float().reshape(*shp[:-1], shp[-1] // G, G)
    return ((q - zp.float().unsqueeze(-1)) * sc.float().unsqueeze(-1)).reshape(shp).to(dtype)

@torch.no_grad()
def gptq(W, H, bits, damp=0.01):
    """Batched GPTQ. W [E,R,C], H [E,C,C] (sum x x^T). returns codes uint8 [E,R,C], scales fp16 [E,R,C/G], zp uint8."""
    E, R, C = W.shape; W = W.float().clone(); H = H.clone()
    qmax = 2 ** bits - 1
    d = torch.diagonal(H, dim1=1, dim2=2)
    dead = d == 0
    idx = torch.arange(C, device=W.device)
    H[:, idx, idx] = torch.where(dead, torch.ones_like(d), d)
    mean = torch.diagonal(H, dim1=1, dim2=2).mean(-1)
    H[:, idx, idx] += (damp * mean)[:, None]
    L = torch.linalg.cholesky(H)
    Hinv = torch.cholesky_inverse(L)
    Hinv = torch.linalg.cholesky(Hinv, upper=True)
    codes = torch.empty(E, R, C, dtype=torch.uint8, device=W.device)
    scs = torch.empty(E, R, C // G, dtype=torch.float16, device=W.device); zps = torch.empty(E, R, C // G, dtype=torch.uint8, device=W.device)
    for gi, i0 in enumerate(range(0, C, G)):
        sc, zp = qparams(W[:, :, i0:i0 + G], bits)       # [E,R,1]
        scs[:, :, gi] = sc.squeeze(-1).half(); zps[:, :, gi] = zp.squeeze(-1).to(torch.uint8)
        Err = torch.zeros(E, R, G, device=W.device)
        for j in range(G):
            w = W[:, :, i0 + j]; dd = Hinv[:, i0 + j, i0 + j]
            q = torch.clamp(torch.round(w / sc[..., 0]) + zp[..., 0], 0, qmax)
            dq = (q - zp[..., 0]) * sc[..., 0]
            codes[:, :, i0 + j] = q.to(torch.uint8)
            err = (w - dq) / dd[:, None]
            W[:, :, i0 + j:i0 + G] -= err[:, :, None] * Hinv[:, i0 + j, None, i0 + j:i0 + G]
            Err[:, :, j] = err
        if i0 + G < C: W[:, :, i0 + G:] -= torch.bmm(Err, Hinv[:, i0:i0 + G, i0 + G:])
    return codes, scs, zps

# ---------------- model ----------------
def rms(x, w, eps):
    xf = x.float(); return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)).to(x.dtype) * w

def rope_cs(pos, hd, theta, dtype):
    inv = 1.0 / (theta ** (torch.arange(0, hd, 2, device=pos.device).float() / hd))
    fr = pos[..., None].float() * inv; emb = torch.cat([fr, fr], -1)
    return emb.cos().to(dtype), emb.sin().to(dtype)

def rot(x, cos, sin):
    h = x.shape[-1] // 2
    return x * cos + torch.cat([-x[..., h:], x[..., :h]], -1) * sin

class Layer:
    """tensors: ln1, ln2, q,k,v,o, qn, kn, gate [K,H], Wgu [K,2I,H], Wd [K,H,I]; ids = original expert ids of the K kept experts"""
    def __init__(self, **kw): self.__dict__.update(kw)

def moe(x2d, L, cfg, valid=None, collect=None):
    if valid is not None:
        idx = valid.nonzero().squeeze(-1); xs = x2d[idx]
    else: xs = x2d
    lg = xs.float() @ L.gate.float().T
    w = torch.softmax(lg, -1); tw, ti = torch.topk(w, cfg["k"], -1)
    if cfg.get("norm_topk", True): tw = tw / tw.sum(-1, keepdim=True)
    Kl = L.gate.shape[0]
    flat = ti.reshape(-1); order = torch.argsort(flat); cnt = torch.bincount(flat, minlength=Kl).tolist()
    tok = order // cfg["k"]; ws = tw.reshape(-1)[order]
    out = torch.zeros_like(xs); off = 0
    if collect is not None: collect.setdefault("mass", torch.zeros(Kl, dtype=torch.float64, device=xs.device)).add_(torch.bincount(flat, weights=tw.reshape(-1).double(), minlength=Kl))
    for e, c in enumerate(cnt):
        if c == 0: continue
        ids = tok[off:off + c]; g = ws[off:off + c]; off += c
        xe = xs[ids]; gu = xe @ L.Wgu[e].T; a, u = gu.chunk(2, -1); h = F.silu(a) * u
        if collect is not None and "hook" in collect: collect["hook"](e, xe, h)
        out.index_add_(0, ids, ((h @ L.Wd[e].T).float() * g[:, None]).to(out.dtype))
    if valid is not None:
        full = torch.zeros_like(x2d); full[idx] = out; return full
    return out

def attn(x, L, cfg, cos, sin, mask, cache=None, ci=None):
    B, T, D = x.shape; nh, nkv, hd = cfg["nh"], cfg["nkv"], cfg["hd"]
    q = (x @ L.q.T).view(B, T, nh, hd); k = (x @ L.k.T).view(B, T, nkv, hd); v = (x @ L.v.T).view(B, T, nkv, hd)
    q = rms(q, L.qn, cfg["eps"]).transpose(1, 2); k = rms(k, L.kn, cfg["eps"]).transpose(1, 2); v = v.transpose(1, 2)
    c, s = cos[:, None], sin[:, None]
    q = rot(q, c, s); k = rot(k, c, s)
    if cache is not None:
        if ci in cache: k = torch.cat([cache[ci][0], k], 2); v = torch.cat([cache[ci][1], v], 2)
        cache[ci] = (k, v)
    r = nh // nkv
    k = k.repeat_interleave(r, 1); v = v.repeat_interleave(r, 1)
    o = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    return o.transpose(1, 2).reshape(B, T, nh * hd) @ L.o.T

def layer_fwd(h, L, cfg, cos, sin, mask, valid=None, cache=None, ci=None, collect=None):
    h = h + attn(rms(h, L.ln1, cfg["eps"]), L, cfg, cos, sin, mask, cache, ci)
    x = rms(h, L.ln2, cfg["eps"]); B, T, D = x.shape
    return h + moe(x.reshape(-1, D), L, cfg, valid.reshape(-1) if valid is not None else None, collect).view(B, T, D)

def causal_mask(am):
    """am [B,T] bool (True = real token), right OR left padded. returns bool [B,1,T,T]"""
    B, T = am.shape
    c = torch.tril(torch.ones(T, T, dtype=torch.bool, device=am.device))
    m = c[None] & am[:, None, :]
    m = m | torch.eye(T, dtype=torch.bool, device=am.device)[None]
    return m[:, None]

class Model:
    def __init__(self, cfg, emb, layers, norm, head, dev):
        self.cfg, self.emb, self.layers, self.norm, self.head, self.dev = cfg, emb, layers, norm, head, dev

    @torch.no_grad()
    def generate(self, ids, am, max_new=8, eos=()):
        """ids/am [B,T] LEFT padded. greedy. returns list of token lists"""
        cfg = self.cfg; B, T = ids.shape; dt = self.emb.dtype
        pos = (am.long().cumsum(-1) - 1).clamp_min(0)
        cos, sin = rope_cs(pos, cfg["hd"], cfg["theta"], dt)
        mask = causal_mask(am); cache = {}
        h = self.emb[ids]
        for i, L in enumerate(self.layers): h = layer_fwd(h, L, cfg, cos, sin, mask, am, cache, i)
        out = [[] for _ in range(B)]; done = torch.zeros(B, dtype=torch.bool, device=ids.device)
        last = rms(h[:, -1], self.norm, cfg["eps"]) @ self.head.T
        nxt = last.argmax(-1)
        curpos = pos[:, -1]; amc = am
        for step in range(max_new):
            for b in range(B):
                if not done[b]: out[b].append(int(nxt[b]))
            done |= torch.isin(nxt, torch.tensor(list(eos), device=ids.device))
            if done.all() or step == max_new - 1: break
            curpos = curpos + 1
            cos, sin = rope_cs(curpos[:, None], cfg["hd"], cfg["theta"], dt)
            amc = torch.cat([amc, torch.ones(B, 1, dtype=torch.bool, device=ids.device)], 1)
            mask = amc[:, None, None, :]
            h = self.emb[nxt[:, None]]
            for i, L in enumerate(self.layers): h = layer_fwd(h, L, cfg, cos, sin, mask, None, cache, i)
            nxt = (rms(h[:, -1], self.norm, cfg["eps"]) @ self.head.T).argmax(-1)
        return out

def cfg_from(hf):
    return dict(nh=hf["num_attention_heads"], nkv=hf["num_key_value_heads"], hd=hf.get("head_dim", hf["hidden_size"] // hf["num_attention_heads"]),
                eps=hf["rms_norm_eps"], theta=hf["rope_theta"], k=hf["num_experts_per_tok"], norm_topk=hf.get("norm_topk_prob", True),
                H=hf["hidden_size"], I=hf["moe_intermediate_size"], NE=hf["num_experts"], NL=hf["num_hidden_layers"], V=hf["vocab_size"])

# ---------------- weight sources ----------------
class ST:
    def __init__(self, d):
        from safetensors import safe_open
        self.d = d; self.idx = json.load(open(f"{d}/model.safetensors.index.json"))["weight_map"]; self.open = {}; self._so = safe_open
    def get(self, name, dev="cpu"):
        f = self.idx[name]
        if f not in self.open: self.open[f] = self._so(f"{self.d}/{f}", "pt", device="cpu")
        return self.open[f].get_tensor(name).to(dev)

def load_layer_fp(st, l, cfg, dev, dt=torch.bfloat16):
    p = f"model.layers.{l}."; g = lambda n: st.get(p + n, dev).to(dt)
    NE = cfg["NE"]
    gu = torch.stack([torch.cat([g(f"mlp.experts.{e}.gate_proj.weight"), g(f"mlp.experts.{e}.up_proj.weight")], 0) for e in range(NE)])
    wd = torch.stack([g(f"mlp.experts.{e}.down_proj.weight") for e in range(NE)])
    return Layer(ln1=g("input_layernorm.weight"), ln2=g("post_attention_layernorm.weight"), q=g("self_attn.q_proj.weight"), k=g("self_attn.k_proj.weight"),
                 v=g("self_attn.v_proj.weight"), o=g("self_attn.o_proj.weight"), qn=g("self_attn.q_norm.weight"), kn=g("self_attn.k_norm.weight"),
                 gate=g("mlp.gate.weight"), Wgu=gu, Wd=wd, ids=list(range(NE)))

def load_base_model(mdir, dev, dt=torch.bfloat16):
    cfg = cfg_from(json.load(open(f"{mdir}/config.json"))); st = ST(mdir)
    layers = [load_layer_fp(st, l, cfg, dev, dt) for l in range(cfg["NL"])]
    return Model(cfg, st.get("model.embed_tokens.weight", dev).to(dt), layers, st.get("model.norm.weight", dev).to(dt), st.get("lm_head.weight", dev).to(dt), dev)

def load_pack_model(pdir, dev, dt=torch.bfloat16):
    """decode a written pack directory (the real artifact) into a runnable Model"""
    from safetensors.torch import load_file
    meta = json.load(open(f"{pdir}/meta.json")); cfg = meta["cfg"]; H, I = cfg["H"], cfg["I"]
    def dq(t, pre, bits, shape):
        n = 1
        for s in shape: n *= s
        codes = unpack_bits(t[pre + ".codes"].to(dev), bits, n).reshape(shape)
        ng = n // G
        zp = unpack_bits(t[pre + ".zp"].to(dev), bits, ng).reshape(*shape[:-1], shape[-1] // G)
        return dequant(codes, t[pre + ".sc"].to(dev), zp, dt)
    e = load_file(f"{pdir}/emb.safetensors"); V = cfg["V"]
    emb = dq(e, "emb", meta["mb"], (V, H)); head = dq(e, "head", meta["hb"], (V, H)); norm = e["norm"].to(dev).to(dt)
    layers = []
    for l in range(cfg["NL"]):
        t = load_file(f"{pdir}/layer{l}.safetensors"); ab = meta["ab"]; qd = cfg["nh"] * cfg["hd"]; kd = cfg["nkv"] * cfg["hd"]
        sh = dict(q=(qd, H), k=(kd, H), v=(kd, H), o=(H, qd))
        a = {n: dq(t, n, ab, sh[n]) for n in sh}
        ids = t["ids"].tolist(); bits = t["ebits"].tolist(); Kl = len(ids)
        Wgu = torch.empty(Kl, 2 * I, H, dtype=dt, device=dev); Wd = torch.empty(Kl, H, I, dtype=dt, device=dev)
        for b in sorted(set(bits)):
            pos = [i for i, x in enumerate(bits) if x == b]; n = len(pos)
            Wgu[pos] = dq(t, f"gu{b}", b, (n, 2 * I, H)); Wd[pos] = dq(t, f"d{b}", b, (n, H, I))
        layers.append(Layer(ln1=t["ln1"].to(dev).to(dt), ln2=t["ln2"].to(dev).to(dt), qn=t["qn"].to(dev).to(dt), kn=t["kn"].to(dev).to(dt),
                            gate=t["gate"].to(dev).to(dt), Wgu=Wgu, Wd=Wd, ids=ids, **a))
    return Model(cfg, emb, layers, norm, head, dev)
