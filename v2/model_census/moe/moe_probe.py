"""MoE census probe (transformers 5.x fused-expert layout; every router returns (logits, weights, indices)).
Stages per model: (1) routing-mass stats on voice-skill routing cal (dev) and NEWS cal; (2) base greedy generation on voice SEL subset; (3) per config: prune experts per layer by voice routing mass,
router masking (recomputed selection among kept experts, same scoring/renorm as the reference), RTN expert quantisation (g128 asym), real greedy generation retention vs base.
Deploy: modal deploy moe_probe.py ; spawn via launch.py. Results: volume tenpct-moe-census:/res/<tag>.json and RESULT_JSON lines in app logs."""
import modal, json, os
app = modal.App("tenpct-moe-census")
HERE = os.path.dirname(os.path.abspath(__file__))
img = (modal.Image.debian_slim(python_version="3.11").pip_install("torch==2.8.0", "transformers==5.22.2", "numpy", "accelerate", "huggingface_hub", "sentencepiece", "protobuf", "tiktoken", "safetensors", "kernels")
       .add_local_file(HERE + "/voice_cal.jsonl", "/root/voice_cal.jsonl").add_local_file(HERE + "/voice_sel.jsonl", "/root/voice_sel.jsonl").add_local_file(HERE + "/news_cal.jsonl", "/root/news_cal.jsonl"))
vol = modal.Volume.from_name("tenpct-moe-census", create_if_missing=True)
hfvol = modal.Volume.from_name("tenpct-v2-hf", create_if_missing=True)

def main(job):
    import time, math, types, collections, torch, torch.nn.functional as F
    from transformers import AutoTokenizer, AutoModelForCausalLM
    T0 = time.time(); log = lambda *a: print(f"[{time.time()-T0:6.0f}s]", *a, flush=True)
    MID = job["model"]; tag = MID.split("/")[-1]; os.environ.setdefault("HF_HOME", "/hf")
    tok = AutoTokenizer.from_pretrained(MID); tok.padding_side = "left"
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    ng = torch.cuda.device_count(); kw = dict(dtype=torch.bfloat16, device_map="auto", max_memory={i: "74GiB" for i in range(ng)})
    if "gpt-oss" in MID:
        from transformers import Mxfp4Config; kw["quantization_config"] = Mxfp4Config(dequantize=True)
    model = AutoModelForCausalLM.from_pretrained(MID, **kw).eval()
    for p in model.parameters(): p.requires_grad_(False)
    log("loaded", sum(p.numel() for p in model.parameters()) / 1e9, "B params")
    # ---- locate MoE blocks
    blocks = []
    for name, mod in model.named_modules():
        ch = dict(mod.named_children())
        if "experts" in ch and ("gate" in ch or "router" in ch):
            ex = ch["experts"]; r = ch.get("gate") or ch["router"]
            if hasattr(ex, "gate_up_proj") and hasattr(ex, "down_proj"): blocks.append((name, mod, r, ex))
            else: log("unsupported experts layout", name, type(ex).__name__)
    assert blocks, "no fused-expert MoE blocks found"
    NL = len(blocks); r0 = blocks[0][2]
    E = getattr(r0, "num_experts", None) or blocks[0][3].gate_up_proj.shape[0]; K = getattr(r0, "top_k", None)
    gptoss = "gpt-oss" in MID
    tot_params = sum(p.numel() for p in model.parameters())
    exp_params = sum(ex.gate_up_proj.numel() + ex.down_proj.numel() for _, _, _, ex in blocks)
    arch = dict(model=MID, moe_layers=NL, experts=E, topk=K, total_params=tot_params, expert_params=exp_params, nonexpert_params=tot_params - exp_params, expert_share=exp_params / tot_params,
                router_cls=type(r0).__name__, experts_cls=type(blocks[0][3]).__name__, gate_up_shape=list(blocks[0][3].gate_up_proj.shape), fp16_gb=tot_params * 2 / 1e9)
    print("ARCH_JSON", json.dumps(arch), flush=True)
    # ---- router wrapping
    STAT = dict(on=False, valid=None); MASK = {}
    mass = [torch.zeros(E, device=r.weight.device if hasattr(r, "weight") else "cuda:0") for _, _, r, _ in blocks]
    cnt = [torch.zeros(E, device=m.device) for m in mass]
    def make_fwd(li, r):
        orig = r.forward; cls = type(r).__name__
        bias_fam = hasattr(r, "e_score_correction_bias") or hasattr(getattr(r, "moe_statics", None), "e_score_correction_bias")
        def fwd(*a, **k):
            out = orig(*a, **k); lg, w, ix = out[0], out[1], out[2]
            m = MASK.get(li)
            if m is not None and not bias_fam:
                lgm = lg.float() + m.to(lg.device)
                if gptoss:
                    tv, ix = torch.topk(lgm, r.top_k, dim=-1); w = torch.softmax(tv, dim=1, dtype=tv.dtype)
                else:
                    pr = torch.softmax(lgm, dim=-1); tv, ix = torch.topk(pr, r.top_k, dim=-1)
                    if cls.startswith("Deepseek"): w = tv * getattr(r, "routed_scaling_factor", 1.0)
                    else: w = tv / tv.sum(-1, keepdim=True)
                w = w.to(lg.dtype); out = (lg, w, ix)
            if STAT["on"]:
                v = STAT["valid"].to(w.device)
                mass[li].index_add_(0, ix.reshape(-1).to(mass[li].device), (w.float() * v[:, None]).reshape(-1).to(mass[li].device))
                cnt[li].index_add_(0, ix.reshape(-1).to(mass[li].device), v[:, None].expand_as(w).reshape(-1).to(mass[li].device))
            return out
        r.forward = fwd
    for li, (_, _, r, _) in enumerate(blocks): make_fwd(li, r)
    # ---- data
    def chat(u):
        kws = dict(tokenize=False, add_generation_prompt=True)
        if gptoss: kws["reasoning_effort"] = "low"
        try: s = tok.apply_chat_template([{"role": "user", "content": u}], enable_thinking=False, **kws)
        except Exception: s = tok.apply_chat_template([{"role": "user", "content": u}], **kws)
        if gptoss: s += "<|channel|>final<|message|>"
        return s
    def load(path, n, step=1):
        rows = [json.loads(l) for l in open(path)][::step][:n]
        for r in rows: r["ids"] = tok(chat(r["user"]), add_special_tokens=False).input_ids
        return rows
    vcal = load("/root/voice_cal.jsonl", 1000); ncal = load("/root/news_cal.jsonl", 600); sel = load("/root/voice_sel.jsonl", job.get("nsel", 700), step=3)
    dev0 = next(model.parameters()).device
    def batches(rows, budget):
        order = sorted(range(len(rows)), key=lambda i: len(rows[i]["ids"])); i = 0
        while i < len(order):
            j = i
            while j < len(order) and (j - i + 1) * len(rows[order[j]]["ids"]) <= budget and (j - i) < 64: j += 1
            j = max(j, i + 1); yield order[i:j]; i = j
    def pad(seqs):
        L = max(len(s) for s in seqs); ids = torch.full((len(seqs), L), tok.pad_token_id); am = torch.zeros((len(seqs), L), dtype=torch.long)
        for r, s in enumerate(seqs): ids[r, L - len(s):] = torch.tensor(s); am[r, L - len(s):] = 1
        return ids.to(dev0), am.to(dev0)
    @torch.no_grad()
    def stats(rows):
        for m, c in zip(mass, cnt): m.zero_(); c.zero_()
        STAT["on"] = True
        for idx in batches(rows, 20000):
            ids, am = pad([rows[k]["ids"] for k in idx]); STAT["valid"] = am.flatten().float().cpu()
            model(input_ids=ids, attention_mask=am, use_cache=False, logits_to_keep=1)
        STAT["on"] = False
        return torch.stack([m.float().cpu() for m in mass]), torch.stack([c.float().cpu() for c in cnt])
    def norm(s): 
        s = s.strip().lower().split("\n")[0].strip(" .,:;\"'`*") if s.strip() else ""
        return s
    @torch.no_grad()
    def gen_eval(rows, max_new=8):
        outs = [None] * len(rows)
        for idx in batches(rows, 30000):
            ids, am = pad([rows[k]["ids"] for k in idx]); L = ids.shape[1]
            g = model.generate(input_ids=ids, attention_mask=am, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.pad_token_id)
            for k, t in zip(idx, tok.batch_decode(g[:, L:], skip_special_tokens=True)): outs[k] = t
        return outs
    # ---- stage 1: routing concentration
    mv, cv = stats(vcal); mn, cn = stats(ncal); log("stats done")
    def conc(m):
        p = m / m.sum(1, keepdim=True); s, _ = p.sort(1, descending=True); cs = s.cumsum(1)
        d = {f"n{int(q*100)}": ((cs < q).sum(1).float() + 1) / E for q in (0.9, 0.95, 0.99)}
        d["entropy_norm"] = -(p.clamp_min(1e-12) * p.clamp_min(1e-12).log()).sum(1) / math.log(E); return d
    cv_, cn_ = conc(mv), conc(mn)
    def captured(msrc, mtgt, f):   # fraction of target routing mass falling in the top-f experts (per layer) of the source
        k = max(1, int(round(f * E))); top = msrc.topk(k, 1).indices; pt = mtgt / mtgt.sum(1, keepdim=True)
        return pt.gather(1, top).sum(1)
    conc_res = dict(voice={k: [round(x, 4) for x in v.tolist()] for k, v in cv_.items()}, news={k: [round(x, 4) for x in v.tolist()] for k, v in cn_.items()},
        summary=dict(voice_n90_mean=cv_["n90"].mean().item(), voice_n95_mean=cv_["n95"].mean().item(), voice_n99_mean=cv_["n99"].mean().item(), news_n90_mean=cn_["n90"].mean().item(),
            voice_entropy_norm=cv_["entropy_norm"].mean().item(), news_entropy_norm=cn_["entropy_norm"].mean().item(),
            voice_mass_in_voice_top50=captured(mv, mv, .5).mean().item(), news_mass_in_voice_top50=captured(mv, mn, .5).mean().item(), news_mass_in_news_top50=captured(mn, mn, .5).mean().item(),
            voice_mass_in_voice_top30=captured(mv, mv, .3).mean().item(), news_mass_in_voice_top30=captured(mv, mn, .3).mean().item(),
            frac_experts_never_used_voice=(cv.sum(0) == 0).float().mean().item() if False else (cv == 0).float().mean().item()))
    print("CONC_JSON", json.dumps(dict(model=MID, **conc_res["summary"])), flush=True)
    res = dict(arch=arch, conc=conc_res, cfgs=[]); os.makedirs("/r/res", exist_ok=True)
    def save(): json.dump(res, open(f"/r/res/{tag}.json", "w")); vol.commit()
    save()
    # ---- stage 2: base generation
    outs0 = gen_eval(sel); bc = [norm(o) == r["answer"].lower() for o, r in zip(outs0, sel)]
    res["base"] = dict(n=len(sel), acc=sum(bc) / len(sel), sample_outs=outs0[:8]); print("BASE_JSON", json.dumps(dict(model=MID, **res["base"])), flush=True); save()
    # ---- stage 3: prune + RTN
    def rtn(w, bits, g):
        sh = w.shape; wg = w.float().reshape(*sh[:-1], sh[-1] // g, g); mx = wg.amax(-1, keepdim=True); mn_ = wg.amin(-1, keepdim=True); qm = 2 ** bits - 1
        sc = ((mx - mn_) / qm).clamp(min=1e-8).half().float(); z = (-mn_ / sc).round().clamp(0, qm)
        return (((wg / sc + z).round().clamp(0, qm) - z) * sc).reshape(sh)
    def pick_g(n):
        for g in (128, 64, 32, 16):
            if n % g == 0: return g
    orig_exp = None
    def snapshot():
        return [(ex.gate_up_proj.data.cpu().clone(), ex.down_proj.data.cpu().clone()) for _, _, _, ex in blocks]
    def bias_t(r):
        if hasattr(r, "e_score_correction_bias"): return r.e_score_correction_bias
        m = getattr(r, "moe_statics", None); return getattr(m, "e_score_correction_bias", None)
    bias0 = {li: bias_t(r).data.clone() for li, (_, _, r, _) in enumerate(blocks) if bias_t(r) is not None}
    keep_snap = job.get("snapshot", exp_params * 2 < 90e9)
    if keep_snap and job["cfgs"]: orig_exp = snapshot(); log("snapshot")
    for cfg in job["cfgs"]:
        t1 = time.time(); f = cfg["f"]; b = cfg.get("bits", 3); k_keep = E if f >= 1 else max(int(round(f * E)), (K or 8))
        if orig_exp is not None:
            for (_, _, _, ex), (g_, d_) in zip(blocks, orig_exp): ex.gate_up_proj.data.copy_(g_.to(ex.gate_up_proj.device)); ex.down_proj.data.copy_(d_.to(ex.down_proj.device))
        for li, b0 in bias0.items(): bias_t(blocks[li][2]).data.copy_(b0)
        for li, (_, _, r, ex) in enumerate(blocks):
            kept = torch.topk(mv[li], k_keep).indices.sort().values; pr = torch.ones(E, dtype=torch.bool); pr[kept] = False
            # router masking
            if f < 1:
                if hasattr(r, "e_score_correction_bias"): 
                    r.e_score_correction_bias.data[pr.to(r.e_score_correction_bias.device)] = -1e4
                elif hasattr(getattr(r, "moe_statics", None), "e_score_correction_bias"):
                    r.moe_statics.e_score_correction_bias.data[..., pr.to(r.moe_statics.e_score_correction_bias.device)] = -1e4
                else: MASK[li] = torch.where(pr, float("-inf"), 0.0).float()
            else: MASK.pop(li, None)
            if b:
                for P, gdim in ((ex.gate_up_proj, 1 if gptoss else -1), (ex.down_proj, 1 if gptoss else -1)):
                    d = P.data; kk = kept.to(d.device)
                    for e in kk.tolist():
                        w = d[e]; w2 = w.transpose(0, 1) if gptoss else w; g = pick_g(w2.shape[-1]); q = rtn(w2, b, g).to(torch.bfloat16)
                        d[e] = (q.transpose(0, 1) if gptoss else q)
        # bytes estimate: kept experts at b bits (+ g128 scale/zero 16+b bits per group) ; non-expert at 6.25 bpw (6-bit g128 incl scale); router rows fp16
        ge = 128; ebits = (b + (16 + b) / ge) if b else 16.0
        est = (exp_params * (k_keep / E) * ebits + (tot_params - exp_params) * 6.25) / 8 + NL * k_keep * model.config.get_text_config().hidden_size * 2
        outs = gen_eval(sel); cor = [norm(o) == r["answer"].lower() for o, r in zip(outs, sel)]
        kb = sum(bc); kk_ = sum(1 for x, y in zip(cor, bc) if x and y)
        cells = collections.defaultdict(lambda: [0, 0])
        for x, y, r in zip(cor, bc, sel):
            if y: c = cells[r["kind"]]; c[0] += 1; c[1] += int(x)
        def wil(k, n, z=1.96):
            if n == 0: return 0.0
            p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); rr = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)); return (c - rr) / d
        cr = dict(name=cfg.get("name", f"f{f}_b{b}"), f=f, bits=b, k_keep=k_keep, est_ratio=est / (tot_params * 2), acc=sum(cor) / len(sel), n_bc=kb, ret=kk_ / max(kb, 1), ret_lo=wil(kk_, kb),
                  worst_cell=min((v[1] / v[0], k_, v[0]) for k_, v in cells.items() if v[0] >= 8) if any(v[0] >= 8 for v in cells.values()) else None, secs=round(time.time() - t1))
        res["cfgs"].append(cr); print("CFG_JSON", json.dumps(dict(model=MID, **cr)), flush=True); save()
        if orig_exp is None: break   # no snapshot: one config per load
    print("RESULT_JSON", json.dumps(dict(model=MID, arch=arch, base=res["base"], conc=conc_res["summary"], cfgs=res["cfgs"])), flush=True); save(); hfvol.commit()

SZ = dict(s=dict(gpu="A100-80GB", memory=128000), m2=dict(gpu="A100-80GB:2", memory=256000), m4=dict(gpu="A100-80GB:4", memory=480000), m8=dict(gpu="A100-80GB:8", memory=900000))
FN = {k: app.function(image=img, timeout=24 * 3600, volumes={"/r": vol, "/hf": hfvol}, env={"HF_HOME": "/hf", "HF_HUB_ENABLE_HF_TRANSFER": "0"}, name="run_" + k, **v)(main) for k, v in SZ.items()}
