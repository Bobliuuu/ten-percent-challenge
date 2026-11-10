"""L12 standalone runtime: loads the packed RTN4 + mass-pruned Qwen3.5-35B-A3B straight from the packed safetensors files (no HF checkpoint, no engine, no dequantised file on disk),
generates greedily, and applies/saves/loads the shared-expert down_proj feedback carrier (fb.bin).
Artifact dir = *.safetensors (44 files from l12_pack.export) + config.json tokenizer.json tokenizer_config.json generation_config.json chat_template.jinja (+ fb.bin/fb.json after feedback).
Forward math is the research engine's MyMoE: router logits of the 64 kept experts, fp32 softmax, top-8, renormalise, bf16 SwiGLU experts, + sigmoid(gate) * shared expert."""
import os, json, glob, math, sys
import torch, torch.nn as nn, torch.nn.functional as F
from safetensors.torch import load_file

NLAY_FB, LAM, AWT, MARGIN, COLS = 10, 2000.0, 30.0, 64.0, 512   # carrier hyperparameters (nlay 10 = dev-only pick of devsel/; REPORT sec 6)

def unpack4(b, n):
    b = b.to(torch.uint8); return torch.stack([b & 15, b >> 4], 1).reshape(-1)[:n]

def dequant(d, name, dev):
    shape = [int(x) for x in d[name + ".shape"]]; n = 1
    for s in shape: n *= s
    sc = d[name + ".sc"].to(dev).float(); ng = sc.numel(); g = n // ng
    c = unpack4(d[name + ".q4"].to(dev), n).float().reshape(ng, g); z = unpack4(d[name + ".z4"].to(dev), ng).float().reshape(ng, 1)
    return ((c - z) * sc.reshape(ng, 1)).reshape(shape)

def names(d): return sorted(k[:-6] for k in d if k.endswith(".shape"))

class PackedMoE(nn.Module):
    def __init__(s, H, K, topk):
        super().__init__(); s.gate = nn.Linear(H, K, bias=False); s.topk = topk; s.shared_expert = None; s.shared_expert_gate = None; s.G = s.U = s.D = None
    def forward(s, hs):
        B, T, D = hs.shape; x = hs.reshape(-1, D)
        p = F.softmax(s.gate(x).float(), -1); K = p.shape[1]
        w, ii = torch.topk(p, min(s.topk, K), -1); w = (w / w.sum(-1, keepdim=True)).to(x.dtype)
        out = torch.zeros_like(x); em = F.one_hot(ii, K).permute(2, 1, 0); hit = (em.sum((1, 2)) > 0).nonzero().flatten().tolist()
        for e in hit:
            k_, t_ = torch.where(em[e]); xe = x[t_]
            a = F.silu(xe @ s.G[e].T) * (xe @ s.U[e].T); y = a @ s.D[e].T; out.index_add_(0, t_, y * w[t_, k_][:, None])
        out = out + torch.sigmoid(s.shared_expert_gate(x)) * s.shared_expert(x)
        return out.view(B, T, D)

def load(adir, dev="cuda", carrier=True):
    import transformers, torch._dynamo   # import before any torch.device('meta') context (lazy model-module import otherwise fails on torch 2.8)
    from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
    import transformers.models.qwen3_5_moe.modeling_qwen3_5_moe  # noqa
    cfg = AutoConfig.from_pretrained(adir); tc = getattr(cfg, "text_config", cfg); dt = torch.bfloat16
    with torch.device("meta"): lm = AutoModelForCausalLM.from_config(tc, dtype=dt, attn_implementation="sdpa")
    K = None
    files = {os.path.basename(f)[:-12]: f for f in glob.glob(os.path.join(adir, "*.safetensors"))}
    K = int(load_file(files["layer0"])["L0.router"].shape[0])
    for lay in lm.model.layers:
        with torch.device("meta"): b = PackedMoE(tc.hidden_size, K, tc.num_experts_per_tok); b.gate = b.gate.to(dt)
        b.shared_expert = lay.mlp.shared_expert; b.shared_expert_gate = lay.mlp.shared_expert_gate; lay.mlp = b
    lm.to_empty(device=dev)
    with torch.device(dev): lm.model.rotary_emb = type(lm.model.rotary_emb)(config=tc)
    P = dict(lm.named_parameters()); seen = set()
    def put(k, t): P[k].data.copy_(t.to(dev, dt)); seen.add(k)
    with torch.no_grad():
        for k, t in load_file(files["small"], device=dev).items(): put(k, t)
        d = load_file(files["attn_shared"])
        for nm in names(d): put(nm, dequant(d, nm, dev))
        for nmx, key in (("embed", "model.embed_tokens.weight"), ("head", "lm_head.weight")):
            d = load_file(files[nmx]); put(key, torch.cat([dequant(d, f"{nmx}.{i}", dev) for i in range(len(names(d)))], 0))
        for l in range(tc.num_hidden_layers):
            d = load_file(files[f"layer{l}"]); blk = lm.model.layers[l].mlp
            put(f"model.layers.{l}.mlp.gate.weight", d[f"L{l}.router"])
            blk.G, blk.U, blk.D = [dequant(d, f"L{l}.{n}", dev).to(dt) for n in ("g", "u", "d")]
    missing = [k for k in P if k not in seen]; assert not missing, missing[:5]
    lm = lm.eval()
    for p in lm.parameters(): p.requires_grad_(False)
    tok = AutoTokenizer.from_pretrained(adir); tok.padding_side = "left"
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    if carrier: load_carrier(lm, adir)
    return lm, tok

class Runner:
    def __init__(s, lm, tok): s.lm, s.tok = lm, tok; s.dev = next(lm.parameters()).device
    def chat(s, u):
        try: return s.tok.apply_chat_template([{"role": "user", "content": u}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except TypeError: return s.tok.apply_chat_template([{"role": "user", "content": u}], tokenize=False, add_generation_prompt=True)
    def encode(s, rows):
        for r in rows:
            if "ids" not in r: r["ids"] = s.tok(s.chat(r["user"]), add_special_tokens=False).input_ids
        return rows
    def batches(s, rows, budget, maxb=96):
        order = sorted(range(len(rows)), key=lambda i: len(rows[i]["ids"])); i = 0
        while i < len(order):
            j = i
            while j < len(order) and (j - i + 1) * len(rows[order[j]]["ids"]) <= budget and (j - i) < maxb: j += 1
            j = max(j, i + 1); yield order[i:j]; i = j
    def pad(s, seqs, left=True):
        L = max(len(x) for x in seqs); ids = torch.full((len(seqs), L), s.tok.pad_token_id); am = torch.zeros((len(seqs), L), dtype=torch.long)
        for r, x in enumerate(seqs):
            if left: ids[r, L - len(x):] = torch.tensor(x); am[r, L - len(x):] = 1
            else: ids[r, :len(x)] = torch.tensor(x); am[r, :len(x)] = 1
        return ids.to(s.dev), am.to(s.dev)
    @torch.no_grad()
    def gen(s, rows, max_new=6, budget=24000):
        """rows: dicts with "user" (full prompt). Same batching/padding/decoding as the engine's gen_eval, so outputs are comparable batch for batch."""
        s.encode(rows); outs = [None] * len(rows)
        for idx in s.batches(rows, budget):
            ids, am = s.pad([rows[k]["ids"] for k in idx]); L = ids.shape[1]
            g = s.lm.generate(input_ids=ids, attention_mask=am, max_new_tokens=max_new, do_sample=False, pad_token_id=s.tok.pad_token_id, temperature=None, top_p=None, top_k=None)
            for k, t in zip(idx, s.tok.batch_decode(g[:, L:], skip_special_tokens=True)): outs[k] = t.strip()
        return outs

# ------------------------------------------------------------------ feedback carrier (port of run_fbf.fb_edit_s / cal_stats)
def _sx(lm, l): return lm.model.layers[l].mlp.shared_expert.down_proj

@torch.no_grad()
def _run_layer(lm, l, h):
    pid = torch.arange(h.shape[1], device=h.device)[None].expand(h.shape[0], -1); pe = lm.model.rotary_emb(h, pid)
    o = lm.model.layers[l](h, attention_mask=None, position_ids=pid, position_embeddings=pe); return o[0] if isinstance(o, tuple) else o

@torch.no_grad()
def _prefix(R, rows, upto):
    ids, am = R.pad([r["ids"] for r in rows], left=False); h = R.lm.model.embed_tokens(ids)
    for l in range(upto): h = _run_layer(R.lm, l, h)
    return h, am

@torch.no_grad()
def cal_stats(R, cal, TL0, ncal=300):
    lm = R.lm; NL = len(lm.model.layers); dev = R.dev; calr = R.encode(cal[:ncal]); hs = []
    for idx in R.batches(calr, 30000, maxb=64):
        sub = [calr[k] for k in idx]; h, am = _prefix(R, sub, TL0); hs.append((h, am.flatten().float(), am.sum(1) - 1))
    Cs = {}; Ca = {}
    for l in range(TL0, NL):
        DI = _sx(lm, l).weight.shape[1]; cs = torch.zeros(DI, DI, device=dev); ca = torch.zeros(DI, DI, device=dev); nt = 0.0; na = 0; cur = {}
        hk1 = _sx(lm, l).register_forward_pre_hook(lambda m, a: cur.__setitem__("k", a[0])); hk2 = lm.model.layers[l].mlp.shared_expert_gate.register_forward_hook(lambda m, a, o: cur.__setitem__("g", torch.sigmoid(o.float())))
        nh = []
        for h, v, lastp in hs:
            o = _run_layer(lm, l, h); k = cur["k"].reshape(-1, DI).float() * cur["g"].reshape(-1, 1) * v[:, None]; cs += k.T @ k; nt += float(v.sum())
            B, T = h.shape[:2]; ar = torch.arange(B, device=dev)
            ka = cur["k"].reshape(B, T, DI)[ar, lastp].float() * cur["g"].reshape(B, T, 1)[ar, lastp]; ca += ka.T @ ka; na += B; nh.append((o, v, lastp))
        hk1.remove(); hk2.remove(); Cs[l] = cs / max(nt, 1.0); Ca[l] = ca / max(na, 1); hs = nh
    return Cs, Ca

@torch.no_grad()
def apply_feedback(R, ann, cal, nlay=NLAY_FB, lam=LAM, awt=AWT, margin=MARGIN, cols=COLS, hsc=None, log=print):
    """ann: [{user, answer}] (<=25). Edits the shared-expert down_proj of the last `nlay` layers in place. Returns (carrier {l:(idx int16, cols bf16 [H,r])}, undo closure)."""
    lm = R.lm; tok = R.tok; dev = R.dev; NL = len(lm.model.layers); TL0 = NL - nlay; R.encode(ann); n = len(ann)
    tgt_ids = [tok(r["answer"], add_special_tokens=False).input_ids[0] for r in ann]
    G = lm.model.norm.weight.float(); eps = getattr(lm.model.norm, "variance_epsilon", getattr(lm.model.norm, "eps", 1e-6)); W = lm.lm_head.weight.float()
    if hsc is None: hsc = cal_stats(R, cal, TL0)
    Cs, Ca = hsc
    h, am = _prefix(R, ann, TL0); last = am.sum(1) - 1; ar = torch.arange(n, device=dev); T = h.shape[1]
    def tail_final(h0, l0):
        hh = h0
        for l in range(l0, NL): hh = _run_layer(lm, l, hh)
        return hh[ar, last].float()
    undo = []; carrier = {}; first = None
    for i, l in enumerate(range(TL0, NL)):
        xf = tail_final(h, l); logits = ((xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)) * G) @ W.T
        rms = xf.pow(2).mean(-1).add(eps).sqrt(); rows = []
        for j in range(n):
            lg = logits[j].clone(); t = tgt_ids[j]; lt = lg[t].item(); lg[t] = -1e9; c = int(lg.argmax()); gap = lg[c].item() - lt
            if gap >= 0: u = G * (W[t] - W[c]); rows.append((j, rms[j] * (gap + margin) * u / (u @ u)))
        if first is None: first = len(rows)
        cur = {}; hk1 = _sx(lm, l).register_forward_pre_hook(lambda m, a: cur.__setitem__("k", a[0])); hk2 = lm.model.layers[l].mlp.shared_expert_gate.register_forward_hook(lambda m, a, o: cur.__setitem__("g", torch.sigmoid(o.float())))
        hn = _run_layer(lm, l, h); hk1.remove(); hk2.remove()
        if rows:
            kk = cur["k"].reshape(n, T, -1)[ar, last].float() * cur["g"].reshape(n, T, 1)[ar, last]; idx = [j for j, _ in rows]
            K = kk[idx]; Rr = torch.stack([d for _, d in rows]) / (nlay - i); DI = K.shape[1]
            B = lam * (Cs[l] + awt * Ca[l]); A = B + K.T @ K + 1e-4 * torch.eye(DI, device=dev) * B.diagonal().mean(); rhs = (Rr.T @ K).T; dW = torch.linalg.solve(A, rhs).T
            top = torch.arange(DI, device=dev)
            if cols < DI: top = dW.norm(dim=0).topk(cols).indices.sort().values; dW = torch.linalg.solve(A[top][:, top], rhs[top]).T
            Wt = _sx(lm, l).weight; old = Wt.data.clone(); Wt.data[:, top] = (Wt.data[:, top].float() + dW.float()).to(Wt.dtype); undo.append((Wt, old))
            carrier[l] = (top.cpu().to(torch.int16), Wt.data[:, top].cpu().clone()); hn = _run_layer(lm, l, h)
        h = hn
    log("annotation sites needing correction:", first)
    def do_undo():
        for Wt, old in undo: Wt.data.copy_(old)
    return carrier, do_undo

def save_carrier(adir, carrier, meta=None):
    with open(os.path.join(adir, "fb.bin"), "wb") as f:
        index = {}
        for l, (idx, cols) in carrier.items():
            a = idx.numpy().tobytes(); b = cols.contiguous().view(torch.int16).numpy().tobytes(); index[str(l)] = [f.tell(), len(idx), cols.shape[0]]; f.write(a); f.write(b)
    json.dump(dict(index=index, meta=meta or {}), open(os.path.join(adir, "fb.json"), "w"))

def clear_carrier(adir):
    for f in ("fb.bin", "fb.json"):
        if os.path.exists(os.path.join(adir, f)): os.remove(os.path.join(adir, f))

def load_carrier(lm, adir):
    p = os.path.join(adir, "fb.json")
    if not os.path.exists(p): return 0
    import numpy as np
    j = json.load(open(p)); raw = open(os.path.join(adir, "fb.bin"), "rb").read(); n = 0
    for l, (off, r, H) in j["index"].items():
        idx = torch.from_numpy(np.frombuffer(raw, dtype=np.int16, count=r, offset=off).astype(np.int64)); cols = torch.from_numpy(np.frombuffer(raw, dtype=np.int16, count=r * H, offset=off + 2 * r).copy()).view(torch.bfloat16).reshape(H, r)
        W = _sx(lm, int(l)).weight; W.data[:, idx.to(W.device)] = cols.to(W.device); n += 1
    return n
