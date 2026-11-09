"""near_miss lane a: expert prune (routing stats on task calib) + RTN low-bit on non-Qwen3 MoE bases. No weight training.
subcommands:  stats BASE | probe MODEL_DIR TAG | build BASE TAG CFGJSON | run BASE CFGLISTJSON
All eval = real greedy generation with vLLM (bf16 weights = dequantised fake-quant of the shipped low-bit tensors)."""
import os, sys, json, math, re, time, glob, shutil, random, subprocess
D = "/workspace/data"; OUT = "/workspace/out"; M = "/workspace/models"
BASES = {
 "olmoe": dict(id="allenai/OLMoE-1B-7B-0125-Instruct"),
 "dsv2l": dict(id="deepseek-ai/DeepSeek-V2-Lite-Chat"),
 "qwen15": dict(id="Qwen/Qwen1.5-MoE-A2.7B-Chat"),
}
EXP_RE = re.compile(r"^(.*\.mlp\.experts\.)(\d+)\.(gate_proj|up_proj|down_proj)\.weight$")
T0 = time.time()
def log(*a): print(f"[{time.time()-T0:6.0f}s]", *a, flush=True)
TOOLP = ("You have access to the following tools (JSON schemas):\n{tools}\n\nIf a tool is needed, call it by replying with one block per call, exactly:\n<tool_call>\n"
         '{{"name": <tool name>, "arguments": {{<argument name>: <value>, ...}}}}\n</tool_call>\nIf no tool is relevant, reply in plain text and do not output any <tool_call> block.\n\nUser request: {q}')

def _unused_t1_prompt(tok, it):
    return tok.apply_chat_template([{"role": "user", "content": TOOLP.format(tools=json.dumps(it["tools"]), q=it["query"])}], add_generation_prompt=True, tokenize=False)
def cls_prompt(tok, k, text):
    P = open(D + ("/tox_prompt.txt" if k == "tox" else "/inj_prompt.txt")).read()
    return tok.apply_chat_template([{"role": "user", "content": P.replace("{t}", text)}], add_generation_prompt=True, tokenize=False)

# ---------------------------------------------------------------- probe (vLLM greedy generation on SKILL7 dev items)
def chat(tok, u): return tok.apply_chat_template([{"role": "user", "content": u}], add_generation_prompt=True, tokenize=False)
def probe(model, tag, max_len=2048):
    from vllm import LLM, SamplingParams
    items = [json.loads(l) for l in open(D + "/skill_dev.jsonl")]
    llm = LLM(model, dtype="bfloat16", max_model_len=max_len, gpu_memory_utilization=0.78, max_num_seqs=128, enforce_eager=False, trust_remote_code=False)
    tok = llm.get_tokenizer(); t0 = time.time()
    o = llm.generate([chat(tok, it["user"]) for it in items], SamplingParams(temperature=0, max_tokens=8), use_tqdm=False)
    R = {"model": tag, "rows": [{"id": it["id"], "src": it["src"], "kind": it["kind"], "gold": it["answer"], "out": o_.outputs[0].text} for it, o_ in zip(items, o)], "secs": time.time() - t0}
    os.makedirs(OUT, exist_ok=True); json.dump(R, open(f"{OUT}/{tag}.json", "w")); print("DONE", tag, R["secs"], flush=True)

# ---------------------------------------------------------------- routing statistics from task calibration prompts (disjoint from dev eval)
def load_hf(base):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from huggingface_hub import snapshot_download
    snap = snapshot_download(BASES[base]["id"], allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.py", "*.tiktoken"])
    tok = AutoTokenizer.from_pretrained(snap)
    lm = AutoModelForCausalLM.from_pretrained(snap, torch_dtype=torch.bfloat16, device_map={"": 0}).eval()
    return snap, tok, lm

def stats(base):
    import torch, torch.nn.functional as F
    p = f"{OUT}/stats_{base}.pt"
    if os.path.exists(p): return torch.load(p)
    snap, tok, lm = load_hf(base)
    cfg = lm.config
    topk = getattr(cfg, "num_experts_per_tok")
    layers = [(i, l.mlp) for i, l in enumerate(lm.model.layers) if hasattr(l.mlp, "experts")]
    NE = len(layers[0][1].experts)
    cnt = {i: torch.zeros(NE, device="cuda") for i, _ in layers}; mass = {i: torch.zeros(NE, device="cuda") for i, _ in layers}; nrm = {i: torch.zeros(NE, device="cuda") for i, _ in layers}
    hs = []
    for i, mlp in layers:
        def gh(mod, inp, out, i=i):
            lg = out[0] if isinstance(out, tuple) else out
            pr = F.softmax(lg.float(), -1); w, ix = torch.topk(pr, topk, -1)
            if getattr(cfg, "norm_topk_prob", False): w = w / w.sum(-1, keepdim=True)
            cnt[i].index_add_(0, ix.reshape(-1), torch.ones_like(w).reshape(-1)); mass[i].index_add_(0, ix.reshape(-1), w.reshape(-1))
        hs.append(mlp.gate.register_forward_hook(gh))
        for j, ex in enumerate(mlp.experts):
            def eh(mod, inp, out, i=i, j=j): nrm[i][j] += out.float().norm(dim=-1).sum()
            hs.append(ex.register_forward_hook(eh))
    items = [chat(tok, json.loads(l)["user"]) for l in open(D + "/skill_cal.jsonl")]
    ntok = 0
    with torch.no_grad():
        for s in items:
            ids = tok(s, return_tensors="pt", add_special_tokens=False).input_ids[:, :3000].cuda(); ntok += ids.shape[1]
            lm(ids, use_cache=False)
    for h in hs: h.remove()
    S = {i: dict(cnt=cnt[i].cpu(), mass=mass[i].cpu(), nrm=nrm[i].cpu()) for i, _ in layers}
    S["_meta"] = dict(ntok=ntok, nitems=len(items), topk=topk, ne=NE)
    os.makedirs(OUT, exist_ok=True); torch.save(S, p); log("stats", base, ntok, "tokens", len(items), "items")
    del lm; torch.cuda.empty_cache(); return S

def score(S, i, crit):
    c, m, n = S[i]["cnt"], S[i]["mass"], S[i]["nrm"]
    if crit == "freq": return c + 1e-6 * m
    if crit == "mass": return m
    if crit == "reap": return (m / c.clamp(min=1)) * (n / c.clamp(min=1)) * (c > 0)      # mean gate x mean expert-output norm over routed tokens (REAP-style approx) 
    if crit == "reapmass": return m * (n / c.clamp(min=1))                               # total gate x norm
    raise ValueError(crit)

# ---------------------------------------------------------------- quantisation + exact byte accounting
def eg(last, g):
    while last % g: g //= 2
    return g
def qbytes(n, bits, g):
    """packed payload + per-group fp16 scale (+ zero-point of `bits` bits unless ternary)"""
    if bits in ("t", 1.58):  return math.ceil(n / 5) + 2 * (n // g)                 # 5 trits per byte, fp16 scale per group
    if bits >= 16: return 2 * n
    return n * bits / 8 + (n // g) * (2 + bits / 8)
def rtn(w, bits, g):
    import torch
    g = eg(w.shape[-1], g)
    sh = w.shape; wg = w.float().reshape(*sh[:-1], sh[-1] // g, g)
    if bits in ("t", 1.58):
        mx = wg.abs().amax(-1, keepdim=True).clamp(min=1e-8); best = None; berr = None
        for r in (0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3):
            sc = (mx * r).half().float(); dq = (wg / sc).round().clamp(-1, 1) * sc; err = ((dq - wg) ** 2).sum(-1, keepdim=True)
            if best is None: best, berr = dq, err
            else: m = err < berr; best = torch.where(m, dq, best); berr = torch.where(m, err, berr)
        return best.reshape(sh)
    mx = wg.amax(-1, keepdim=True); mn = wg.amin(-1, keepdim=True); qm = 2 ** bits - 1
    def q(mn_, mx_):
        sc = ((mx_ - mn_) / qm).clamp(min=1e-8).half().float(); z = (-mn_ / sc).round().clamp(0, qm)
        return ((wg / sc + z).round().clamp(0, qm) - z) * sc
    best = None; berr = None
    for r in (1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6):
        dq = q(mn * r, mx * r); err = ((dq - wg) ** 2).sum(-1, keepdim=True)
        if best is None: best, berr = dq, err
        else: m = err < berr; best = torch.where(m, dq, best); berr = torch.where(m, err, berr)
    return best.reshape(sh)

def classify(name, shape):
    m = EXP_RE.match(name)
    if m: return "expert", int(m.group(2))
    if name.endswith("mlp.gate.weight"): return "router", None
    if len(shape) == 2 and ("embed_tokens" in name or "lm_head" in name): return "embed", None
    if len(shape) == 2 and min(shape) >= 16: return "attn", None
    return "fp16", None

def plan_bytes(shapes, cfg, K, ne, aux_bytes):
    """shapes: {name: shape}. returns total shipped bytes and breakdown."""
    bd = dict(expert=0, attn=0, embed=0, router=0, fp16=0, aux=aux_bytes); nlay = set()
    for n, sh in shapes.items():
        k, j = classify(n, sh); numel = 1
        for d in sh: numel *= d
        if k == "expert":
            if j == 0: nlay.add(n.split(".mlp.")[0])
            if j < K: bd["expert"] += qbytes(numel, cfg["xb"], eg(sh[-1], cfg["xg"]))
        elif k == "router": bd["router"] += 2 * numel * K // ne
        elif k == "embed": bd["embed"] += qbytes(numel, cfg["eb"], eg(sh[-1], cfg["eg"]))
        elif k == "attn": bd["attn"] += qbytes(numel, cfg["ab"], eg(sh[-1], cfg["ag"]))
        else: bd["fp16"] += 2 * numel
    bd["index"] = len(nlay) * K      # kept-expert ids (1 byte each)
    return sum(bd.values()), bd

def pick_K(shapes, cfg, ne, aux, fp16_bytes, frac=0.099):
    best = 0
    for K in range(1, ne + 1):
        if plan_bytes(shapes, cfg, K, ne, aux)[0] <= frac * fp16_bytes: best = K
    return best

def get_shapes(snap):
    from safetensors import safe_open
    sh = {}
    for f in glob.glob(snap + "/*.safetensors"):
        with safe_open(f, "pt") as g:
            for k in g.keys(): sh[k] = tuple(g.get_slice(k).get_shape())
    return sh

def aux_bytes(snap):
    t = 0
    for f in ["tokenizer.json", "tokenizer_config.json", "config.json", "generation_config.json", "vocab.json", "merges.txt", "special_tokens_map.json", "tokenizer.model"]:
        p = os.path.join(snap, f)
        if os.path.exists(p): t += os.path.getsize(p)
    return t + 30000      # + loader/metadata code allowance

def build(base, tag, cfg):
    """cfg: xb,xg (experts bits/group), ab,ag (attn+shared+dense), eb,eg (embed+lm_head), crit, K (or 'max'), norm (bool: norm_topk_prob)"""
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file
    from huggingface_hub import snapshot_download
    snap = snapshot_download(BASES[base]["id"], allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.py", "*.tiktoken"])
    shapes = get_shapes(snap); fp16_bytes = 2 * sum(math.prod(s) for s in shapes.values())
    config = json.load(open(snap + "/config.json"))
    ne = config.get("num_experts") or config.get("n_routed_experts"); aux = aux_bytes(snap)
    K = cfg["K"] if cfg["K"] != "max" else pick_K(shapes, cfg, ne, aux, fp16_bytes)
    total, bd = plan_bytes(shapes, cfg, K, ne, aux)
    S = stats(base) if K < ne else None
    nl_key = lambda n: int(re.search(r"layers\.(\d+)\.", n).group(1))
    kept = {}
    if S is not None:
        for i in S:
            if i == "_meta": continue
            kept[i] = sorted(torch.topk(score(S, i, cfg.get("crit", "mass")), K).indices.tolist())
    new_id = {i: {o: n for n, o in enumerate(kept[i])} for i in kept}
    out = {}; dev = "cuda"
    files = sorted(glob.glob(snap + "/*.safetensors"))
    for f in files:
        with safe_open(f, "pt") as g:
            for k in g.keys():
                kind, j = classify(k, shapes[k]); li = nl_key(k) if "layers." in k else None
                if kind == "expert":
                    if K < ne and j not in new_id[li]: continue
                    nk = EXP_RE.match(k); k2 = f"{nk.group(1)}{new_id[li][j] if K < ne else j}.{nk.group(3)}.weight"
                    w = g.get_tensor(k).to(dev); out[k2] = (rtn(w, cfg["xb"], cfg["xg"]) if (cfg["xb"] == "t" or cfg["xb"] < 16) else w.float()).to(torch.bfloat16).cpu()
                elif kind == "router":
                    w = g.get_tensor(k)
                    out[k] = w[kept[li]].contiguous() if K < ne else w
                elif kind == "embed":
                    w = g.get_tensor(k).to(dev); out[k] = (rtn(w, cfg["eb"], cfg["eg"]) if cfg["eb"] < 16 else w.float()).to(torch.bfloat16).cpu()
                elif kind == "attn":
                    w = g.get_tensor(k).to(dev); out[k] = (rtn(w, cfg["ab"], cfg["ag"]) if cfg["ab"] < 16 else w.float()).to(torch.bfloat16).cpu()
                else: out[k] = g.get_tensor(k)
    d = f"{M}/{tag}"; shutil.rmtree(d, ignore_errors=True); os.makedirs(d)
    # shard
    keys = list(out); sh = []; cur = {}; sz = 0
    for k in keys:
        cur[k] = out[k].contiguous(); sz += cur[k].numel() * 2
        if sz > 4e9: sh.append(cur); cur = {}; sz = 0
    if cur: sh.append(cur)
    wm = {}
    for n, s in enumerate(sh):
        fn = f"model-{n:05d}.safetensors"; save_file(s, f"{d}/{fn}", metadata={"format": "pt"}); wm.update({k: fn for k in s})
    json.dump({"metadata": {}, "weight_map": wm}, open(d + "/model.safetensors.index.json", "w"))
    config = dict(config)
    if K < ne:
        if "num_experts" in config: config["num_experts"] = K
        if "n_routed_experts" in config: config["n_routed_experts"] = K
        for kk in ("num_experts_per_tok",):
            config[kk] = min(config[kk], K)
        if config.get("topk_method") == "group_limited_greedy": config["topk_method"] = "greedy"; config["n_group"] = 1; config["topk_group"] = 1
    if cfg.get("norm"): config["norm_topk_prob"] = True
    config.pop("auto_map", None); config.pop("quantization_config", None)
    json.dump(config, open(d + "/config.json", "w"), indent=1)
    for f in os.listdir(snap):
        if f.endswith((".json", ".txt", ".model", ".tiktoken")) and f not in ("config.json", "model.safetensors.index.json") and os.path.isfile(snap + "/" + f): shutil.copy(snap + "/" + f, d + "/" + f)
    torch.cuda.empty_cache()
    info = dict(tag=tag, base=base, cfg=cfg, K=K, ne=ne, fp16_bytes=fp16_bytes, shipped_bytes=total, ratio=total / fp16_bytes, breakdown=bd)
    json.dump(info, open(f"{OUT}/{tag}.size.json", "w"), indent=1); log("BUILT", json.dumps(info))
    return info

def run(base, cfgs):
    for tag, cfg in cfgs:
        if os.path.exists(f"{OUT}/{tag}.json"): continue
        try:
            info = build(base, tag, cfg)
            r = subprocess.run([sys.executable, __file__, "probe", f"{M}/{tag}", tag], capture_output=True, text=True)
            print(r.stdout[-600:], r.stderr[-1500:] if r.returncode else "", flush=True)
        except Exception as e:
            import traceback; traceback.print_exc()
        shutil.rmtree(f"{M}/{tag}", ignore_errors=True)


# ---------------------------------------------------------------- GPTQ (GRAY: weight update from calibration Hessians; no gradients/training) on the pruned model
def gptq_q(W, H, bits, g, damp=0.01, blk=128):
    g = eg(W.shape[1], g)
    """W: (out,in) float cuda; H: (in,in) float cuda. column-wise OBS update, group asym minmax (fp16 scale, int zero)."""
    import torch
    W = W.clone().float(); n = W.shape[1]; qm = 2 ** bits - 1
    H = H.clone(); dead = H.diagonal() == 0; H[dead, dead] = 1; W[:, dead] = 0
    H += damp * H.diagonal().mean() * torch.eye(n, device=W.device)
    try: Hi = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(H)), upper=True)
    except Exception: return rtn(W, bits, g)
    Q = torch.zeros_like(W)
    for i1 in range(0, n, blk):
        i2 = min(i1 + blk, n); W1 = W[:, i1:i2].clone(); Q1 = torch.zeros_like(W1); E1 = torch.zeros_like(W1); Hi1 = Hi[i1:i2, i1:i2]
        for i in range(i2 - i1):
            if (i1 + i) % g == 0:
                grp = W1[:, i:i + g]
                mx = grp.amax(1, keepdim=True); mn = grp.amin(1, keepdim=True)
                sc = ((mx - mn) / qm).clamp(min=1e-8).half().float(); z = (-mn / sc).round().clamp(0, qm)
            w = W1[:, i:i + 1]; q = ((w / sc + z).round().clamp(0, qm) - z) * sc
            Q1[:, i:i + 1] = q; e = (w - q) / Hi1[i, i]; W1[:, i:] -= e @ Hi1[i:i + 1, i:]; E1[:, i:i + 1] = e
        Q[:, i1:i2] = Q1; W[:, i2:] -= E1 @ Hi[i1:i2, i2:]
    return Q

def gptq_qb(W, H, bits, g, damp=0.01, blk=128):
    """batched GPTQ. W: (B,out,in) float cuda; H: (B,in,in). column-wise OBS update; group asym minmax (fp16 scale, int zero)."""
    import torch
    g = eg(W.shape[2], g); W = W.clone().float(); B, o, n = W.shape; qm = 2 ** bits - 1; I = torch.eye(n, device=W.device)
    H = H.clone(); dd = H.diagonal(dim1=1, dim2=2); dead = dd == 0
    H = H + torch.diag_embed(dead.float()); W = W.masked_fill(dead[:, None, :], 0)
    dm = H.diagonal(dim1=1, dim2=2).mean(1)
    for dmp in (damp, 0.1, 1.0):
        Hd = H + (dmp * dm)[:, None, None] * I
        L, info = torch.linalg.cholesky_ex(Hd)
        if int(info.max()) == 0: break
    else: return torch.stack([rtn(W[b], bits, g) for b in range(B)])
    Hi = torch.linalg.cholesky(torch.cholesky_inverse(L), upper=True)
    Q = torch.zeros_like(W)
    for i1 in range(0, n, blk):
        i2 = min(i1 + blk, n); W1 = W[:, :, i1:i2].clone(); Q1 = torch.zeros_like(W1); E1 = torch.zeros_like(W1); Hi1 = Hi[:, i1:i2, i1:i2]
        for i in range(i2 - i1):
            if (i1 + i) % g == 0:
                grp = W1[:, :, i:i + g]; mx = grp.amax(2, keepdim=True); mn = grp.amin(2, keepdim=True)
                sc = ((mx - mn) / qm).clamp(min=1e-8).half().float(); z = (-mn / sc).round().clamp(0, qm)
            w = W1[:, :, i:i + 1]; q = ((w / sc + z).round().clamp(0, qm) - z) * sc
            Q1[:, :, i:i + 1] = q; e = (w - q) / Hi1[:, i, i][:, None, None]; W1[:, :, i:] -= e @ Hi1[:, i:i + 1, i:]; E1[:, :, i:i + 1] = e
        Q[:, :, i1:i2] = Q1; W[:, :, i2:] -= E1 @ Hi[:, i1:i2, i2:]
    return Q

def gptq_build(base, tag, cfg):
    """pruned fp model -> Hessians from task calibration prompts (EASY calib, all tasks, disjoint from dev) -> GPTQ experts, RTN attn/embed -> save"""
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from huggingface_hub import snapshot_download
    snap = snapshot_download(BASES[base]["id"], allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.py", "*.tiktoken"])
    shapes = get_shapes(snap); fp16_bytes = 2 * sum(math.prod(s) for s in shapes.values()); ne = (json.load(open(snap + "/config.json")).get("num_experts") or json.load(open(snap + "/config.json")).get("n_routed_experts"))
    K = cfg["K"] if cfg["K"] != "max" else pick_K(shapes, cfg, ne, aux_bytes(snap), fp16_bytes)
    cfg = dict(cfg); cfg["K"] = K
    pre = dict(cfg); pre.update(xb=16, ab=16, eb=16); ptag = "pre_" + tag
    info = build(base, ptag, pre); d = f"{M}/{ptag}"
    tot, bd = plan_bytes(shapes, cfg, K, ne, aux_bytes(snap)); info.update(shipped_bytes=tot, ratio=tot / fp16_bytes, breakdown=bd)
    tok = AutoTokenizer.from_pretrained(d); lm = AutoModelForCausalLM.from_pretrained(d, torch_dtype=torch.bfloat16, device_map={"": 0}).eval()
    layers = [(i, l.mlp) for i, l in enumerate(lm.model.layers) if hasattr(l.mlp, "experts")]
    Hg = {}; Hd = {}; hs = []
    for i, mlp in layers:
        for j, ex in enumerate(mlp.experts):
            def hg(mod, inp, i=i, j=j):
                x = inp[0].float().reshape(-1, inp[0].shape[-1]); Hg[(i, j)] = Hg.get((i, j), 0) + x.T @ x
            def hd(mod, inp, i=i, j=j):
                x = inp[0].float().reshape(-1, inp[0].shape[-1]); Hd[(i, j)] = Hd.get((i, j), 0) + x.T @ x
            hs.append(ex.gate_proj.register_forward_pre_hook(hg)); hs.append(ex.down_proj.register_forward_pre_hook(hd))
    cal = [json.loads(l) for l in open(D + "/calib_all.jsonl")]
    with torch.no_grad():
        for it in cal:
            ids = tok(chat(tok, it["user"]), return_tensors="pt", add_special_tokens=False).input_ids[:, :1500].cuda(); lm(ids, use_cache=False)
    for h in hs: h.remove()
    log("hessians done", len(Hg))
    with torch.no_grad():
        for i, mlp in layers:
            ne_ = len(mlp.experts); Hgs = torch.stack([Hg.get((i, j), torch.eye(mlp.experts[j].gate_proj.weight.shape[1], device="cuda")) for j in range(ne_)]); Hds = torch.stack([Hd.get((i, j), torch.eye(mlp.experts[j].down_proj.weight.shape[1], device="cuda")) for j in range(ne_)])
            Wgu = torch.cat([torch.stack([mlp.experts[j].gate_proj.weight.data.float() for j in range(ne_)]), torch.stack([mlp.experts[j].up_proj.weight.data.float() for j in range(ne_)])], 1)
            Qgu = gptq_qb(Wgu, Hgs, cfg["xb"], cfg["xg"]); oo = Qgu.shape[1] // 2
            Qd = gptq_qb(torch.stack([mlp.experts[j].down_proj.weight.data.float() for j in range(ne_)]), Hds, cfg["xb"], cfg["xg"])
            for j in range(ne_):
                ex = mlp.experts[j]; ex.gate_proj.weight.data = Qgu[j, :oo].to(torch.bfloat16); ex.up_proj.weight.data = Qgu[j, oo:].to(torch.bfloat16); ex.down_proj.weight.data = Qd[j].to(torch.bfloat16)
            for j in range(ne_): Hg.pop((i, j), None); Hd.pop((i, j), None)
            del Hgs, Hds, Wgu, Qgu, Qd; torch.cuda.empty_cache(); log("gptq layer", i)
        Hg.clear(); Hd.clear(); torch.cuda.empty_cache()
        for n_, p in lm.named_parameters():
            k, _ = classify(n_, tuple(p.shape))
            if k == "attn": p.data = rtn(p.data.float(), cfg["ab"], cfg["ag"]).to(torch.bfloat16)
            elif k == "embed": p.data = rtn(p.data.float(), cfg["eb"], cfg["eg"]).to(torch.bfloat16)
    d2 = f"{M}/{tag}"; shutil.rmtree(d2, ignore_errors=True); lm.save_pretrained(d2, safe_serialization=True); tok.save_pretrained(d2)
    del lm; torch.cuda.empty_cache(); shutil.rmtree(d, ignore_errors=True)
    info2 = dict(info); info2.update(tag=tag, cfg=cfg); json.dump(info2, open(f"{OUT}/{tag}.size.json", "w"), indent=1); log("GPTQ BUILT", tag, info2["ratio"]); return info2

def rungptq(base, cfgs):
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    for tag, cfg in cfgs:
        if os.path.exists(f"{OUT}/{tag}.json"): continue
        try:
            subprocess.run([sys.executable, __file__, "gptqbuild", base, tag, json.dumps(cfg)], env=os.environ, check=True)
            r = subprocess.run([sys.executable, __file__, "probe", f"{M}/{tag}", tag], capture_output=True, text=True, env=env)
            print(r.stdout[-300:], r.stderr[-1500:] if r.returncode else "", flush=True)
        except Exception:
            import traceback; traceback.print_exc()
        shutil.rmtree(f"{M}/{tag}", ignore_errors=True)

if __name__ == "__main__":
    c = sys.argv[1]
    if c == "probe": probe(sys.argv[2], sys.argv[3])
    elif c == "baseprobe":
        from huggingface_hub import snapshot_download
        probe(snapshot_download(BASES[sys.argv[2]]["id"], allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.py", "*.tiktoken"]), "base_" + sys.argv[2])
    elif c == "stats": stats(sys.argv[2])
    elif c == "run": run(sys.argv[2], json.load(open(sys.argv[3])))
    elif c == "gptqbuild": gptq_build(sys.argv[2], sys.argv[3], json.loads(sys.argv[4]))
    elif c == "rungptq": rungptq(sys.argv[2], json.load(open(sys.argv[3])))
    elif c == "build": build(sys.argv[2], sys.argv[3], json.loads(sys.argv[4]))
