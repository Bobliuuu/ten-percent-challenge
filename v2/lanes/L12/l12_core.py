"""L12 MoE-combination engine (simulated/fake-quant, no training, no gradients).
Qwen3-30B-A3B-Instruct-2507 (E=128 routed experts, top-8, no shared expert). Expert originals live as stacked CPU tensors read straight from the
safetensors checkpoint (version independent); the GPU holds the non-expert model + the CONFIGURED experts only.
A config (dict) composes: score -> kept set (uniform / global water-fill / KL-layer-sensitivity) -> optional merge of pruned experts into kept ones
(functional-similarity clustering, optional neuron-permutation alignment, usage-weighted average, router mass summed over the cluster) ->
optional shared-subspace coding of the kept experts (mean + shared right basis + low-bit coefficients) -> RTN or sequential batched GPTQ(+Hadamard)
-> top-k override.  Bytes are accounted analytically from the configured bit-widths (all shipped parts).  Real greedy generation eval."""
import os, json, math, time, glob, collections, types, hashlib, sys
import torch, torch.nn as nn, torch.nn.functional as F

T0 = time.time()
def log(*a): print(f"[{time.time()-T0:6.0f}s]", *a, flush=True)

def wilson(k, n, z=1.96):
    if n == 0: return (0.0, 0.0)
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); r = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - r) / d, (c + r) / d)

# ---------------------------------------------------------------- quantizers
def bpw(bits, g): return 16.0 if not bits else bits + (16 + bits) / g   # fp16 scale + b-bit zero per group

def rtn(w, bits, g, mse=True):
    """asymmetric per-group RTN along last dim; fp16 scale, integer zero. returns dequantised float tensor (same shape)"""
    sh = w.shape; g = math.gcd(sh[-1], g); wg = w.float().reshape(*sh[:-1], sh[-1] // g, g); mx = wg.amax(-1, keepdim=True); mn = wg.amin(-1, keepdim=True); qm = 2 ** bits - 1
    def q(mn_, mx_):
        sc = ((mx_ - mn_) / qm).clamp(min=1e-8).half().float(); z = (-mn_ / sc).round().clamp(0, qm)
        return ((wg / sc + z).round().clamp(0, qm) - z) * sc
    if not mse: return q(mn, mx).reshape(sh)
    best = None; berr = None
    for r in (1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65):
        dq = q(mn * r, mx * r); err = ((dq - wg) ** 2).sum(-1, keepdim=True)
        if best is None: best, berr = dq, err
        else: m = err < berr; best = torch.where(m, dq, best); berr = torch.where(m, err, berr)
    return best.reshape(sh)

def had(n):
    Hm = torch.ones(1, 1)
    while Hm.shape[0] < n: Hm = torch.cat([torch.cat([Hm, Hm], 1), torch.cat([Hm, -Hm], 1)], 0)
    return Hm / math.sqrt(n)

def rot_matrix(dim, blk, seed, dev):
    g = torch.Generator().manual_seed(seed); sg = (torch.randint(0, 2, (dim,), generator=g) * 2 - 1).float()
    B = had(blk); Q = torch.zeros(dim, dim)
    for s0 in range(0, dim, blk): Q[s0:s0 + blk, s0:s0 + blk] = B
    return (sg[:, None] * Q).to(dev)   # orthogonal

@torch.no_grad()
def gptq_b(W, H, bits, g, damp=0.01):
    """batched GPTQ. W [B,o,i], H [B,i,i] (sum x x^T, fp32). asym per-group g along i (group fixed when reached), fp16 scale + int zero. returns dequantised [B,o,i]"""
    W = W.float().clone(); H = H.float().clone(); B, o, i_ = W.shape; qm = 2 ** bits - 1
    d = torch.diagonal(H, dim1=1, dim2=2); dead = d == 0
    if dead.any():
        bi, ii = dead.nonzero(as_tuple=True); H[bi, ii, ii] = 1; W[bi, :, ii] = 0
    mean = torch.diagonal(H, dim1=1, dim2=2).mean(1)
    eye = torch.eye(i_, device=W.device)
    for dm in (damp, damp * 10, damp * 100):
        Hd = H + (dm * mean)[:, None, None] * eye
        L, info = torch.linalg.cholesky_ex(Hd)
        if (info == 0).all(): break
    ok = info == 0
    if not ok.all(): Hd[~ok] = eye * mean[~ok][:, None, None]; L = torch.linalg.cholesky(Hd)
    Hi = torch.linalg.cholesky(torch.cholesky_inverse(L), upper=True)
    Q = torch.zeros_like(W)
    for b0 in range(0, i_, g):
        W1 = W[:, :, b0:b0 + g].clone(); Hi1 = Hi[:, b0:b0 + g, b0:b0 + g]; E = torch.zeros_like(W1)
        mn = W1.amin(2, keepdim=True); mx = W1.amax(2, keepdim=True); sc = ((mx - mn) / qm).clamp(min=1e-8).half().float(); z = (-mn / sc).round().clamp(0, qm)
        for j in range(W1.shape[2]):
            w = W1[:, :, j:j + 1]; q = ((w / sc + z).round().clamp(0, qm) - z) * sc; Q[:, :, b0 + j:b0 + j + 1] = q
            err = (w - q) / Hi1[:, j, j][:, None, None]; W1[:, :, j:] -= err @ Hi1[:, j:j + 1, j:]; E[:, :, j:j + 1] = err
        if b0 + g < i_: W[:, :, b0 + g:] -= E @ Hi[:, b0:b0 + g, b0 + g:]
    return Q

# ---------------------------------------------------------------- MoE block
class Live:
    """per-layer live expert state. G,U [K,I,H], D [K,H,I] stacked; loc [E] local idx or -1; M [E,K] cluster mass-sum matrix or None"""
    def __init__(s): s.G = s.U = s.D = None; s.keptidx = None; s.M = None; s.off = None

class MyMoE(nn.Module):
    def __init__(s, H, E, topk):
        super().__init__(); s.gate = nn.Linear(H, E, bias=False); s.E = E; s.topk = topk; s.shared_expert = None; s.shared_expert_gate = None; s.live = Live(); s.rec = None; s.valid = None; s.hess = None; s.samp = None; s.capture = None
    def forward(s, hs):
        B, T, D = hs.shape; x = hs.reshape(-1, D); lv = s.live
        p = F.softmax(s.gate(x).float(), -1)
        pk = (p @ lv.M) if lv.M is not None else p[:, lv.keptidx]
        if lv.off is not None: pk = pk.clone(); pk[:, lv.off] = -1.0
        K = pk.shape[1]; w, ii = torch.topk(pk, min(s.topk, K), -1); w = (w / w.sum(-1, keepdim=True)).to(x.dtype)
        out = torch.zeros_like(x); valid = s.valid
        em = F.one_hot(ii, K).permute(2, 1, 0); hit = (em.sum((1, 2)) > 0).nonzero().flatten().tolist()
        for e in hit:
            k_, t_ = torch.where(em[e]); xe = x[t_]
            a = F.silu(xe @ lv.G[e].T) * (xe @ lv.U[e].T); y = a @ lv.D[e].T; we = w[t_, k_]
            out.index_add_(0, t_, y * we[:, None])
            if s.capture is not None:
                pm = s.capture["mask"][t_]
                if pm.any(): s.capture["rec"].append((e, t_[pm], we[pm].float(), a[pm].float()))
            if s.rec is not None or s.hess is not None:
                v = valid[t_] if valid is not None else torch.ones(len(t_), device=x.device)
                if s.rec is not None:
                    yn = y.float().norm(dim=-1)
                    s.rec[0][e] += v.sum(); s.rec[1][e] += (we.float() * v).sum(); s.rec[2][e] += (we.float() * yn * v).sum(); s.rec[3][e] += (yn * v).sum()
                if s.hess is not None:
                    vb = v.bool(); xv = xe[vb].float(); av = a[vb].float(); h = s.hess; h[0][e].addmm_(xv.T, xv); h[1][e].addmm_(av.T, av); h[2][e] += xv.shape[0]
        if s.samp is not None and valid is not None:
            vi = valid.nonzero().flatten()
            if len(vi): s.samp.append(x[vi[torch.randperm(len(vi), device=x.device)[:s.samp_n]]].float().cpu())
        if s.shared_expert is not None: out = out + torch.sigmoid(s.shared_expert_gate(x)) * s.shared_expert(x)
        return out.view(B, T, D)

# ---------------------------------------------------------------- engine
class Engine:
    def __init__(s, model_id, data_dir, out_dir, dtype=torch.bfloat16, dev="cuda", snap=None):
        from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
        from huggingface_hub import snapshot_download
        import safetensors
        s.dev = dev; s.dtype = dtype; s.out = out_dir; os.makedirs(out_dir + "/res", exist_ok=True); s.model_id = model_id
        s.snap = snap or snapshot_download(model_id, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.jinja"])
        s.cfgm = AutoConfig.from_pretrained(s.snap); tc = getattr(s.cfgm, "text_config", s.cfgm)
        s.tc = tc; s.NL = tc.num_hidden_layers; s.E = tc.num_experts; s.TOPK = tc.num_experts_per_tok; s.HID = tc.hidden_size; s.FF = tc.moe_intermediate_size
        s.tok = AutoTokenizer.from_pretrained(s.snap); s.tok.padding_side = "left"
        s.idx = json.load(open(s.snap + "/model.safetensors.index.json"))["weight_map"]; s._fh = {}
        s._sf = safetensors
        s.q35 = hasattr(s.cfgm, "text_config")
        with torch.device("meta"): lm = AutoModelForCausalLM.from_config(tc if s.q35 else s.cfgm, dtype=dtype, attn_implementation="sdpa")
        s.lm = lm; s.blocks = []
        for l, lay in enumerate(lm.model.layers):
            with torch.device("meta"): b = MyMoE(s.HID, s.E, s.TOPK)
            if getattr(lay.mlp, "shared_expert", None) is not None: b.shared_expert = lay.mlp.shared_expert; b.shared_expert_gate = lay.mlp.shared_expert_gate
            lay.mlp = b; s.blocks.append(b)
        lm.to_empty(device=dev)
        with torch.device(dev): rot = type(lm.model.rotary_emb)(config=tc)
        lm.model.rotary_emb = rot
        s.nonexp_numel = 0; s.attn2d = 0; s.small = 0; s.shared_n = 0
        sd = {k: v for k, v in lm.named_parameters()}
        for k, p in sd.items():
            t = s.T(k); p.data.copy_(t.to(dev, dtype))
            if (".self_attn." in k or ".linear_attn." in k) and k.endswith(".weight") and p.dim() == 2: s.attn2d += p.numel()
            elif ".shared_expert." in k and p.dim() == 2: s.shared_n += p.numel()
            elif "embed_tokens" in k or "lm_head" in k or k.endswith("mlp.gate.weight"): pass
            else: s.small += p.numel()
        s.embed_n = lm.model.embed_tokens.weight.numel(); s.head_n = lm.lm_head.weight.numel()
        s.orig_attn = {k: s.T(k) for k, p in sd.items() if (".self_attn." in k or ".linear_attn." in k) and k.endswith(".weight") and p.dim() == 2}   # cpu originals for requant
        s.orig_shared = {k: s.T(k) for k, p in sd.items() if ".shared_expert." in k and p.dim() == 2}
        s.orig_emb = lm.model.embed_tokens.weight.data.clone().cpu(); s.orig_head = lm.lm_head.weight.data.clone().cpu()
        log("non-expert model on", dev, "attn2d", s.attn2d, "embed", s.embed_n, "head", s.head_n)
        # stacked expert originals on CPU
        s.OG = []; s.OU = []; s.OD = []
        for l in range(s.NL):
            if s.q35:
                gu = s.T(f"model.layers.{l}.mlp.experts.gate_up_proj").to(dtype); g = gu[:, :s.FF].contiguous(); u = gu[:, s.FF:].contiguous(); d = s.T(f"model.layers.{l}.mlp.experts.down_proj").to(dtype); del gu
            else:
                g = torch.stack([s.T(f"model.layers.{l}.mlp.experts.{e}.gate_proj.weight") for e in range(s.E)]).to(dtype)
                u = torch.stack([s.T(f"model.layers.{l}.mlp.experts.{e}.up_proj.weight") for e in range(s.E)]).to(dtype)
                d = torch.stack([s.T(f"model.layers.{l}.mlp.experts.{e}.down_proj.weight") for e in range(s.E)]).to(dtype)
            s.OG.append(g); s.OU.append(u); s.OD.append(d)
            if l % 8 == 0: log("read experts layer", l)
        s.orig_gate = [b.gate.weight.data.clone() for b in s.blocks]
        s.expert_params = s.NL * s.E * 3 * s.FF * s.HID
        s.total_params = s.expert_params + s.nonexp_params()
        s.fp16_bytes = int(json.load(open(s.snap + "/model.safetensors.index.json"))["metadata"]["total_size"]); log("param check", s.total_params * 2, s.fp16_bytes)
        s.fp16_text = s.total_params * 2 if s.q35 else s.fp16_bytes
        s.tokb = sum(os.path.getsize(os.path.join(s.snap, f)) for f in ["tokenizer.json", "tokenizer_config.json", "config.json", "generation_config.json"] if os.path.exists(os.path.join(s.snap, f)))
        log("total params", s.total_params, "fp16 bytes", s.fp16_bytes, "tok bytes", s.tokb)
        s.load_data(data_dir)
        s.attn_state = None; s.stats = None
    def nonexp_params(s): return s.attn2d + s.small + s.shared_n + s.embed_n + s.head_n + s.NL * s.E * s.HID   # gate rows counted as non-expert
    def T(s, key):
        if getattr(s, "q35", False) and key.startswith("model."): key = "model.language_model." + key[6:]
        f = s.idx[key]
        if f not in s._fh: s._fh[f] = s._sf.safe_open(os.path.join(s.snap, f), "pt", device="cpu")
        return s._fh[f].get_tensor(key)
    # ------------------------------------------------ data
    def chat(s, u):
        try: return s.tok.apply_chat_template([{"role": "user", "content": u}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except TypeError: return s.tok.apply_chat_template([{"role": "user", "content": u}], tokenize=False, add_generation_prompt=True)
    def load_data(s, d):
        s.D = {}
        for n in ("cal", "sel", "rep"):
            rows = [json.loads(l) for l in open(f"{d}/{n}.jsonl")]
            for r in rows: r["ids"] = s.tok(s.chat(r["user"]), add_special_tokens=False).input_ids
            s.D[n] = rows
        log("data", {k: len(v) for k, v in s.D.items()})
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
    def gen_eval(s, rows, budget=24000, max_new=None):
        max_new = max_new or getattr(s, 'max_new', 6)
        outs = [None] * len(rows)
        for idx in s.batches(rows, budget):
            ids, am = s.pad([rows[k]["ids"] for k in idx]); L = ids.shape[1]
            g = s.lm.generate(input_ids=ids, attention_mask=am, max_new_tokens=max_new, do_sample=False, pad_token_id=s.tok.pad_token_id, temperature=None, top_p=None, top_k=None)
            for k, t in zip(idx, s.tok.batch_decode(g[:, L:], skip_special_tokens=True)): outs[k] = t.strip()
        return outs
    # ------------------------------------------------ live helpers
    @torch.no_grad()
    def set_full(s, l):
        lv = s.blocks[l].live; lv.G = s.OG[l].to(s.dev); lv.U = s.OU[l].to(s.dev); lv.D = s.OD[l].to(s.dev)
        lv.keptidx = torch.arange(s.E, device=s.dev); lv.M = None; lv.off = None
        s.blocks[l].gate.weight.data.copy_(s.orig_gate[l]); s.blocks[l].topk = s.TOPK
    def set_full_all(s):
        for l in range(s.NL): s.set_full(l)
        torch.cuda.empty_cache()
    def free_live(s):
        for b in s.blocks: b.live = Live()
        torch.cuda.empty_cache()
    def restore_nonexpert(s):
        PP = dict(s.lm.named_parameters())
        for k, t in list(s.orig_attn.items()) + list(s.orig_shared.items()): PP[k].data.copy_(t.to(s.dev, s.dtype))
        s.lm.model.embed_tokens.weight.data.copy_(s.orig_emb.to(s.dev)); s.lm.lm_head.weight.data.copy_(s.orig_head.to(s.dev))
    def quant_nonexpert(s, cfg):
        s.restore_nonexpert(); P = dict(s.lm.named_parameters())
        if cfg.get("attn_b"):
            for k in s.orig_attn: P[k].data.copy_(rtn(P[k].data, cfg["attn_b"], cfg.get("attn_g", 128), True).to(s.dtype))
        if cfg.get("shared_b"):
            for k in s.orig_shared: P[k].data.copy_(rtn(P[k].data, cfg["shared_b"], 128, True).to(s.dtype))
        for tgt, b in ((s.lm.model.embed_tokens, cfg.get("emb_b")), (s.lm.lm_head, cfg.get("head_b"))):
            if b:
                for s0 in range(0, tgt.weight.shape[0], 20000): tgt.weight.data[s0:s0 + 20000] = rtn(tgt.weight.data[s0:s0 + 20000], b, 128, False).to(s.dtype)
    # ------------------------------------------------ calibration statistics (original model)
    @torch.no_grad()
    def run_stats(s, ncal=None, nsamp=1024):
        p = f"{s.out}/res/stats_{ncal}.pt"
        if os.path.exists(p): s.stats = torch.load(p); return s.stats
        rows = s.D["cal"][:ncal]; s.set_full_all(); s.restore_nonexpert()
        rec = [[torch.zeros(s.E, device=s.dev) for _ in range(4)] for _ in range(s.NL)]
        for l, b in enumerate(s.blocks): b.rec = rec[l]; b.samp = []; b.samp_n = 64
        for idx in s.batches(rows, 30000):
            ids, am = s.pad([rows[k]["ids"] for k in idx]); v = am.flatten().float()
            for b in s.blocks: b.valid = v
            s.lm.model(input_ids=ids, attention_mask=am, use_cache=False)
        xs = []
        for b in s.blocks:
            x = torch.cat(b.samp); g = torch.Generator().manual_seed(0); x = x[torch.randperm(len(x), generator=g)[:nsamp]]; xs.append(x.half()); b.rec = None; b.samp = None; b.valid = None
        st = dict(cnt=torch.stack([r[0] for r in rec]).cpu(), mass=torch.stack([r[1] for r in rec]).cpu(), reap_sum=torch.stack([r[2] for r in rec]).cpu(), nrm_sum=torch.stack([r[3] for r in rec]).cpu(), xs=torch.stack(xs))
        torch.save(st, p); s.stats = st; log("stats done", len(rows)); return st
    # ------------------------------------------------ layer runner (for sequential GPTQ and KL-LOO)
    def rope(s, h, T):
        pid = torch.arange(T, device=s.dev)[None].expand(h.shape[0], -1); return pid, s.lm.model.rotary_emb(h, pid)
    @torch.no_grad()
    def run_layer(s, l, h):
        pid, pe = s.rope(h, h.shape[1]); o = s.lm.model.layers[l](h, attention_mask=None, position_ids=pid, position_embeddings=pe)
        return o[0] if isinstance(o, tuple) else o
    @torch.no_grad()
    def loo_scores(s, topn=48, nprompt=24):
        """KL leave-one-expert-out at the answer position: KL(base || ablated) for the top-`topn` experts by mass in each layer (others 0)."""
        p = f"{s.out}/res/loo_{topn}_{nprompt}.pt"
        if os.path.exists(p): return torch.load(p)
        st = s.stats; rows = s.D["cal"][-nprompt:]; s.set_full_all(); s.restore_nonexpert()
        ids, am = s.pad([r["ids"] for r in rows], left=False); last = am.sum(1) - 1; ar = torch.arange(len(rows), device=s.dev)
        h = s.lm.model.embed_tokens(ids); Hin = []
        for l in range(s.NL): Hin.append(h); h = s.run_layer(l, h)
        def head(h): return F.log_softmax(s.lm.lm_head(s.lm.model.norm(h[ar, last])).float(), -1)
        base = head(h); S = torch.zeros(s.NL, s.E)
        for l in range(s.NL):
            cand = torch.topk(st["mass"][l], min(topn, s.E)).indices.tolist(); cand = [e for e in cand if st["cnt"][l, e] > 0]
            for e in cand:
                s.blocks[l].live.off = torch.tensor([e], device=s.dev); h = Hin[l]
                for j in range(l, s.NL): h = s.run_layer(j, h)
                S[l, e] = F.kl_div(head(h), base, log_target=True, reduction="batchmean").item()
            s.blocks[l].live.off = None
            if l % 6 == 0: log("loo layer", l)
        torch.save(S, p); return S
    @torch.no_grad()
    def layer_sens(s, frac=0.5, nprompt=48):
        """KL when pruning the bottom `frac` (by mass) experts of ONE layer."""
        p = f"{s.out}/res/lsens_{frac}_{nprompt}.pt"
        if os.path.exists(p): return torch.load(p)
        st = s.stats; rows = s.D["cal"][-nprompt:]; s.set_full_all(); s.restore_nonexpert()
        ids, am = s.pad([r["ids"] for r in rows], left=False); last = am.sum(1) - 1; ar = torch.arange(len(rows), device=s.dev)
        h = s.lm.model.embed_tokens(ids); Hin = []
        for l in range(s.NL): Hin.append(h); h = s.run_layer(l, h)
        def head(h): return F.log_softmax(s.lm.lm_head(s.lm.model.norm(h[ar, last])).float(), -1)
        base = head(h); S = torch.zeros(s.NL)
        for l in range(s.NL):
            drop = torch.topk(st["mass"][l], int(s.E * frac), largest=False).indices.to(s.dev); s.blocks[l].live.off = drop; h = Hin[l]
            for j in range(l, s.NL): h = s.run_layer(j, h)
            S[l] = F.kl_div(head(h), base, log_target=True, reduction="batchmean").item(); s.blocks[l].live.off = None
        torch.save(S, p); return S
    # ------------------------------------------------ scores / allocation
    def score(s, name):
        st = s.stats; cnt = st["cnt"]; mass = st["mass"]
        if name == "freq": return cnt.clone()
        if name == "mass": return mass.clone()                                  # task-specific "unified" score: routing frequency x gate weight
        if name == "reap": return st["reap_sum"] / cnt.clamp(min=1)              # REAP: mean gate*||expert out|| per routed token
        if name == "reapsum": return st["reap_sum"].clone()                      # total contribution = mass x mean ||y||
        if name == "gatefree": return st["nrm_sum"] / cnt.clamp(min=1)           # task-agnostic gate-free routed-token-averaged ||y||
        if name == "rand": g = torch.Generator().manual_seed(1); return torch.rand(s.NL, s.E, generator=g)
        if name.startswith("kl"):
            S = s.loo_scores(); return S + 1e-9 * mass / mass.sum(1, keepdim=True)
        raise ValueError(name)
    def kept_sets(s, cfg):
        sc = s.score(cfg.get("score", "reap")); keep = cfg.get("keep", 1.0); E = s.E; NL = s.NL
        if cfg.get("kept_counts"): kc = cfg["kept_counts"]
        else:
            k = int(round(keep * E)) if keep <= 1.0 else int(keep); ad = cfg.get("adapt", "uniform"); tot = k * NL; kmin = cfg.get("kmin", 8)
            if ad == "uniform": kc = [k] * NL
            elif ad == "global":      # water-fill on per-layer-normalised score
                fr = sc / sc.sum(1, keepdim=True).clamp(min=1e-12); kc = [kmin] * NL; keepset = [set(torch.topk(fr[l], kmin).indices.tolist()) for l in range(NL)]
                cand = sorted(((fr[l, e].item(), l, e) for l in range(NL) for e in range(E) if e not in keepset[l]), reverse=True)[:tot - kmin * NL]
                for _, l, e in cand: kc[l] += 1
            elif ad == "cover":       # per-layer count to cover X of routing mass, scaled to the global total
                m = s.stats["mass"]; fr = m / m.sum(1, keepdim=True); cs = torch.sort(fr, 1, descending=True).values.cumsum(1)
                need = [(cs[l] < cfg.get("cover", 0.95)).sum().item() + 1 for l in range(NL)]; scale = tot / sum(need); kc = [max(kmin, min(E, int(round(n * scale)))) for n in need]
            elif ad == "kl":          # counts proportional to layer KL sensitivity ^ gamma
                S = s.layer_sens().clamp(min=1e-6) ** cfg.get("gamma", 0.5); w = S / S.sum() * tot; kc = [max(kmin, min(E, int(round(x)))) for x in w.tolist()]
            else: raise ValueError(ad)
            d = tot - sum(kc); i = 0
            while d != 0 and i < 100000:
                l = i % NL; step = 1 if d > 0 else -1
                if kmin <= kc[l] + step <= E: kc[l] += step; d -= step
                i += 1
        kept = [sorted(torch.topk(sc[l], kc[l]).indices.tolist()) for l in range(NL)]
        return kept
    # ------------------------------------------------ merging
    @torch.no_grad()
    def merge_layer(s, l, kept, cfg, G, U, D):
        """G,U,D: all-expert originals on GPU [E,...]. returns merged stacked (Gk,Uk,Dk) for kept, cluster matrix M [E,K], plus info"""
        E = s.E; K = len(kept); kt = torch.tensor(kept, device=s.dev); pr = [e for e in range(E) if e not in set(kept)]
        mass = s.stats["mass"][l].to(s.dev); x = s.stats["xs"][l].to(s.dev).to(s.dtype)
        a = F.silu(torch.einsum("sh,eih->esi", x, G)) * torch.einsum("sh,eih->esi", x, U)      # [E,S,I]
        y = torch.einsum("esi,ehi->esh", a, D).float().reshape(E, -1); yn = F.normalize(y, dim=-1); sim = yn @ yn.T
        Gk, Uk, Dk = G[kt].clone().float(), U[kt].clone().float(), D[kt].clone().float()
        sim_kp = sim[:, kt]; assign = {}; M = torch.zeros(E, K, device=s.dev); M[kt, torch.arange(K, device=s.dev)] = 1
        wsum = mass[kt] * cfg.get("merge_self", 1.0) + 1e-6; acc_g = Gk * wsum[:, None, None]; acc_u = Uk * wsum[:, None, None]; acc_d = Dk * wsum[:, None, None]
        thr = cfg.get("merge_min_mass", 0.0); tot = mass.sum()
        for p_ in pr:
            ki = int(sim_kp[p_].argmax()); assign[p_] = ki
            if mass[p_] / tot < thr: M[p_, ki] = 1; continue          # negligible usage: route mass but do not average weights
            M[p_, ki] = 1; wp = float(mass[p_]) + 1e-6
            gp, up_, dp = G[p_].float(), U[p_].float(), D[p_].float()
            if cfg.get("merge", "avg") == "align":
                ak = (a[kt[ki]].float()); ap = a[p_].float()
                ak = F.normalize(ak - ak.mean(0), dim=0); ap = F.normalize(ap - ap.mean(0), dim=0); C = (ak.T @ ap).cpu().numpy()   # [I_k, I_p]
                from scipy.optimize import linear_sum_assignment
                r, c = linear_sum_assignment(-C); perm = torch.tensor(c, device=s.dev)         # kept neuron r <- pruned neuron perm[r]
                gp, up_, dp = gp[perm], up_[perm], dp[:, perm]
            acc_g[ki] += wp * gp; acc_u[ki] += wp * up_; acc_d[ki] += wp * dp; wsum[ki] += wp
        Gk = acc_g / wsum[:, None, None]; Uk = acc_u / wsum[:, None, None]; Dk = acc_d / wsum[:, None, None]
        if not cfg.get("merge_router", True): M = None
        return Gk.to(s.dtype), Uk.to(s.dtype), Dk.to(s.dtype), M
    # ------------------------------------------------ shared subspace
    @torch.no_grad()
    def subspace(s, Wk, r, cb, bb, gb=128, gc=64, delta=True):
        """Wk [K,o,i]. W_e ~ Wbar + C_e V^T; Wbar = mean (quantised bb bits), V [i,r] shared right basis (fp16), C_e = (W_e-Wbar) V quantised cb bits (group gc along r).
        returns recon [K,o,i], bytes"""
        K, o, i = Wk.shape; Wf = Wk.float(); Wbar = Wf.mean(0) if delta else torch.zeros(o, i, device=Wk.device)
        Wb = rtn(Wbar, bb, gb, True) if delta and bb else Wbar
        dl = Wf - Wb[None]; Gm = dl.reshape(K * o, i).T @ dl.reshape(K * o, i)
        ev, evec = torch.linalg.eigh(Gm); V = evec[:, -r:].flip(1)        # [i,r]
        C = dl @ V                                                    # [K,o,r]
        Cq = rtn(C, cb, gc, True) if cb else C
        rec = Wb[None] + Cq @ V.T
        nb = (o * i * bpw(bb, gb) / 8 if delta else 0) + i * r * 2 + K * o * r * bpw(cb, gc) / 8
        return rec.to(Wk.dtype), nb
    # ------------------------------------------------ build
    @torch.no_grad()
    def build(s, cfg):
        """apply cfg to the live model. returns kept lists and bytes dict"""
        t1 = time.time(); kept = s.kept_sets(cfg); s.free_live(); s.quant_nonexpert(cfg); E = s.E; NL = s.NL; HID, FF = s.HID, s.FF
        eb, eg = cfg.get("eb"), cfg.get("eg", 128); quant = cfg.get("quant", "rtn"); sub = cfg.get("subspace"); merge = cfg.get("merge")
        nbytes = collections.Counter(); seq = bool(eb) and quant == "gptq"
        if seq:
            rows = s.D["cal"][:cfg.get("nh", 1000)]; hs = []
            for idx in s.batches(rows, 30000, maxb=64):
                ids, am = s.pad([rows[k]["ids"] for k in idx], left=False); hs.append((s.lm.model.embed_tokens(ids), am.flatten().float()))
        fb = 0
        for l in range(NL):
            K = len(kept[l]); kt = torch.tensor(kept[l], device=s.dev); blk = s.blocks[l]; lv = blk.live
            G, U, D = s.OG[l].to(s.dev), s.OU[l].to(s.dev), s.OD[l].to(s.dev); M = None
            if merge: Gk, Uk, Dk, M = s.merge_layer(l, kept[l], cfg, G, U, D)
            else: Gk, Uk, Dk = G[kt], U[kt], D[kt]
            del G, U, D
            lv.keptidx = kt; lv.M = M
            if M is not None: nbytes["router"] += E * HID * 2; nbytes["cluster_map"] += E
            else: nbytes["router"] += K * HID * 2; nbytes["keep_mask"] += E / 8
            if cfg.get("topk"): blk.topk = cfg["topk"]
            else: blk.topk = s.TOPK
            if sub:
                r_in, r_dn = sub["r"], sub.get("r_down", sub["r"] // 2)
                for nm, W in (("g", Gk), ("u", Uk)):
                    rec, nb = s.subspace(W, r_in, sub["cb"], sub.get("bb", 4), delta=sub.get("delta", True)); nbytes["experts"] += nb
                    if nm == "g": Gk = rec
                    else: Uk = rec
                Dk, nb = s.subspace(Dk, min(r_dn, FF), sub["cb"], sub.get("bb", 4), delta=sub.get("delta", True)); nbytes["experts"] += nb
            elif eb and quant != "gptq":
                Gk, Uk, Dk = [rtn(W, eb, eg, True).to(s.dtype) for W in (Gk, Uk, Dk)]; nbytes["experts"] += K * 3 * FF * HID * bpw(eb, eg) / 8
            elif not eb: nbytes["experts"] += K * 3 * FF * HID * 2
            lv.G, lv.U, lv.D = Gk, Uk, Dk
            if seq:
                nh = blk.hess = [torch.zeros(K, HID, HID, device=s.dev), torch.zeros(K, FF, FF, device=s.dev), torch.zeros(K, device=s.dev)]
                for h, v in hs: blk.valid = v; s.run_layer(l, h)
                blk.hess = None; blk.valid = None; Hn = nh
                QI = rot_matrix(HID, min(256, HID), 1000 + l, s.dev) if cfg.get("had", True) else torch.eye(HID, device=s.dev)
                QD = rot_matrix(FF, min(256, FF), 2000 + l, s.dev) if cfg.get("had", True) else torch.eye(FF, device=s.dev)
                nmin = cfg.get("nmin", 32); bad = Hn[2] < nmin; fb += int(bad.sum())
                Hg = QI.T @ Hn[0] @ QI; Hd = QD.T @ Hn[1] @ QD; gu = torch.cat([Gk, Uk], 1).float() @ QI; dd = Dk.float() @ QD
                outg = torch.empty_like(gu); outd = torch.empty_like(dd); cs = cfg.get("gchunk", 32)
                for c0 in range(0, K, cs):
                    sl = slice(c0, c0 + cs); outg[sl] = gptq_b(gu[sl], Hg[sl], eb, eg, cfg.get("damp", 0.01)); outd[sl] = gptq_b(dd[sl], Hd[sl], eb, eg, cfg.get("damp", 0.01))
                qg = outg @ QI.T; qd = outd @ QD.T
                if bad.any():
                    rg = rtn(torch.cat([Gk, Uk], 1), eb, eg, True); rd = rtn(Dk, eb, eg, True); qg[bad] = rg[bad]; qd[bad] = rd[bad]
                lv.G, lv.U, lv.D = qg[:, :FF].to(s.dtype), qg[:, FF:].to(s.dtype), qd.to(s.dtype); nbytes["experts"] += K * 3 * FF * HID * bpw(eb, eg) / 8
                del Hn, nh, Hg, Hd, gu, dd, outg, outd
                hs = [(s.run_layer(l, h), v) for h, v in hs]
            if l % 8 == 0: log("built layer", l, "K", K)
        # non-expert bytes
        nbytes["attn"] = s.attn2d * bpw(cfg.get("attn_b"), cfg.get("attn_g", 128)) / 8
        nbytes["embed"] = s.embed_n * bpw(cfg.get("emb_b"), 128) / 8; nbytes["head"] = s.head_n * bpw(cfg.get("head_b"), 128) / 8
        nbytes["shared_expert"] = s.shared_n * bpw(cfg.get("shared_b"), 128) / 8; nbytes["norms_misc"] = s.small * 2; nbytes["tokenizer_config"] = s.tokb; nbytes["code_meta"] = cfg.get("code_bytes", 60000)
        tot = sum(nbytes.values()); by = dict(parts={k: int(v) for k, v in nbytes.items()}, total=int(tot), ratio=tot / s.fp16_bytes, ratio_text_only=tot / s.fp16_text, fp16_bytes=s.fp16_bytes, fp16_text_bytes=s.fp16_text)
        torch.cuda.empty_cache(); log("build done", cfg.get("name"), "ratio", round(by["ratio"], 4), "rtn fallbacks", fb, f"{time.time()-t1:.0f}s")
        return kept, by, fb
    # ------------------------------------------------ eval
    def metrics(s, rows, outs, base):
        corr = [o == r["answer"] for o, r in zip(outs, rows)]; m = dict(n=len(rows), acc=sum(corr) / len(rows))
        if base is None: return m
        bc = [o == r["answer"] for o, r in zip(base, rows)]; nb = sum(bc); kk = sum(1 for x, y in zip(corr, bc) if x and y)
        cells = collections.defaultdict(lambda: [0, 0]); ag = sum(1 for a_, b_ in zip(outs, base) if a_ == b_)
        for x, y, r in zip(corr, bc, rows):
            if y: c = cells[r["src"] + "|" + r["kind"]]; c[0] += 1; c[1] += int(x)
        cl = {k: dict(n=v[0], k=v[1], ret=v[1] / v[0], lo=wilson(v[1], v[0])[0]) for k, v in sorted(cells.items())}
        bad = collections.Counter(o for x, y, o in zip(corr, bc, outs) if y and not x)
        m.update(base_acc=nb / len(rows), n_bc=nb, kept_bc=kk, ret=kk / max(nb, 1), ret_lo=wilson(kk, nb)[0], agree=ag / len(rows), cells=cl,
                 min_cell=min([v["ret"] for v in cl.values()] or [0]), min_cell_lo=min([v["lo"] for v in cl.values()] or [0]), flips=dict(bad.most_common(6)))
        return m
    def run_cfg(s, cfg):
        name = cfg["name"]; fp = f"{s.out}/res/{name}.json"
        if os.path.exists(fp) and not cfg.get("force"): log("skip", name); return json.load(open(fp))
        t1 = time.time()
        if name == "base":
            s.free_live(); s.set_full_all(); s.restore_nonexpert(); kept = [list(range(s.E))] * s.NL; by = dict(total=s.fp16_bytes, ratio=1.0, parts={}); fb = 0
        else: kept, by, fb = s.build(cfg)
        res = dict(name=name, cfg=cfg, bytes=by, kept_per_layer=[len(k) for k in kept], rtn_fallbacks=fb, sets={})
        for sn in cfg.get("sets", ["sel", "rep"]):
            rows = s.D[sn]; outs = s.gen_eval(rows); bp = f"{s.out}/res/base_{sn}.json"
            if name == "base": json.dump(dict(outs=outs, ids=[r["id"] for r in rows]), open(bp, "w"))
            base = json.load(open(bp))["outs"] if os.path.exists(bp) else None
            res["sets"][sn] = s.metrics(rows, outs, base); json.dump(dict(outs=outs), open(f"{s.out}/res/outs_{name}_{sn}.json", "w"))
        res["secs"] = round(time.time() - t1)
        json.dump(res, open(fp, "w")); sm = res["sets"].get("sel", {})
        print("RESULT", name, json.dumps(dict(ratio=round(by["ratio"], 4), acc=sm.get("acc"), ret=sm.get("ret"), lo=sm.get("ret_lo"), min_cell=sm.get("min_cell"))), flush=True)
        return res

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--cfgs", required=True); ap.add_argument("--model", default="Qwen/Qwen3-30B-A3B-Instruct-2507"); ap.add_argument("--data", default="data"); ap.add_argument("--out", default="out")
    ap.add_argument("--ncal", type=int, default=1231); ap.add_argument("--only", default=None); ap.add_argument("--max_new", type=int, default=6)
    a = ap.parse_args()
    cfgs = json.load(open(a.cfgs)); eng = Engine(a.model, a.data, a.out); eng.max_new = a.max_new
    eng.run_cfg(dict(name="base", sets=["sel", "rep"])); eng.run_stats(a.ncal)
    for cfg in cfgs:
        if a.only and a.only not in cfg["name"]: continue
        try: eng.run_cfg(cfg)
        except Exception as ex:
            import traceback; traceback.print_exc(); print("FAILED", cfg["name"], repr(ex)[:300], flush=True); eng.free_live(); torch.cuda.empty_cache()
