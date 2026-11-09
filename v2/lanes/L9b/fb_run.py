"""L9b feedback on the PACKED model: diffuse closed-form edit through the (dense, every-layer) shared-expert down_proj of the last MoE layers, column-sparse fp16 carrier.
Mechanism (no optimiser, no new tensor): teacher-force each annotated (prompt, label); at answer positions where argmax != label token, closed-form residual target
dx = rms(x) (gap+m) u/(u.u), u = (1+norm.w)*(W_U[t]-W_U[c]); ascending over layers, layer l delivers 1/(layers left) of the CURRENT remaining dx via shared_expert.down_proj
(keys = sigmoid(shared_expert_gate) * down_proj input): dW = R^T K (lam (C + awt C_ans + ridge) + K^T K)^-1, then restricted to the r columns of largest norm and re-solved there.
Carrier = the r replaced fp16 columns per edited layer + int16 indices (stored as fb.bin next to the artifact, counted). Controls: empty / real / shuffled / wrong labels."""
import json, os, random, math, collections, time
import torch
import core

def run(job, model, tok, R, S, man, log, commit):
    cfgs = job["fb"]; dev = R.dev0; EOS = "<|im_end|>"
    layers = model.model.layers; NLAY = len(layers); H = model.config.hidden_size
    WU = model.lm_head.weight.float(); G = (1.0 + model.model.norm.weight.float()); eps = model.config.rms_norm_eps
    tgt_ids = lambda lab: tok(lab + EOS, add_special_tokens=False).input_ids
    CAP = dict(on=False); hooks = []
    def tmod(l, target):
        """module whose weight is edited: shared-expert down_proj (gated keys) or the attention output projection (o_proj on full-attention layers, out_proj on Gated-DeltaNet layers)."""
        if target == "shared_down": return layers[l].mlp.shared_expert.down_proj
        return layers[l].self_attn.o_proj if hasattr(layers[l], "self_attn") else layers[l].linear_attn.out_proj
    def install(LAY, target):
        for h in hooks: h.remove()
        hooks.clear()
        for l in LAY:
            hooks.append(tmod(l, target).register_forward_pre_hook(lambda m, a, l=l: CAP.__setitem__(("k", l), a[0].detach().reshape(-1, a[0].shape[-1])) if CAP["on"] else None))
            if target == "shared_down": hooks.append(layers[l].mlp.shared_expert_gate.register_forward_hook(lambda m, a, o, l=l: CAP.__setitem__(("g", l), torch.sigmoid(o.detach().float())) if CAP["on"] else None))
        hooks.append(model.model.norm.register_forward_pre_hook(lambda m, a: CAP.__setitem__("fin", a[0].detach()) if CAP["on"] else None))
    def fwd(seqs):
        torch.cuda.empty_cache(); Lm = max(len(s) for s in seqs); ids = torch.full((len(seqs), Lm), tok.pad_token_id); am = torch.zeros((len(seqs), Lm), dtype=torch.long)
        for r, s in enumerate(seqs): ids[r, :len(s)] = torch.tensor(s); am[r, :len(s)] = 1
        CAP["on"] = True; model.model(input_ids=ids.to(dev), attention_mask=am.to(dev), use_cache=False); CAP["on"] = False; return Lm
    def kall(l, lo, hi):
        k = CAP[("k", l)][lo:hi].float(); g = CAP.get(("g", l)); return k * g[lo:hi].float() if g is not None else k
    def keys(l, b, p, Lm): return kall(l, b * Lm + p, b * Lm + p + 1)[0].to(dev)
    cal = S["CAL"][:job.get("ncovs", 300)]; R.encode(cal)
    def cov(LAY, target):
        DI = tmod(LAY[0], target).weight.shape[1]
        Cs = {l: torch.zeros(DI, DI, device=dev) for l in LAY}; Ca = {l: torch.zeros(DI, DI, device=dev) for l in LAY}; nt = na = 0
        for s0 in range(0, len(cal), 16):
            bt = cal[s0:s0 + 16]; seqs = [r["ids"] + tgt_ids(r["answer"]) for r in bt]; Lm = fwd(seqs)
            for l in LAY:
                for b, r in enumerate(bt):
                    k = kall(l, b * Lm, b * Lm + len(seqs[b])); Cs[l] += k.T @ k; ka = k[len(r["ids"]) - 1:]; Ca[l] += ka.T @ ka
            nt += sum(len(s) for s in seqs); na += sum(len(tgt_ids(r["answer"])) for r in bt)
        for l in LAY: Cs[l] /= nt; Ca[l] /= na
        return Cs, Ca
    pre = job.get("prefix", "skill7"); NEW = job["new_label"]
    DEV = job.get("dev", False); seed0 = job.get("seed0", 500 if DEV else 100)
    if DEV:   # config selection data ONLY: annotations from pool[:200], dev teach = pool[200:] (never annotated), dev neighbours = SEL items (selection split) the packed model answers correctly; nothing locked is touched
        so = R.gen(S["SEL"]); sc_ = core.correct(so, S["SEL"]); log("SEL packed-model accuracy (identity check vs eval_s7_k102_e3.json comp_acc):", sum(sc_) / len(sc_), sum(sc_), len(sc_))
        neigh = [(r, o) for r, o, c in zip(S["SEL"], so, sc_) if c][::job.get("sel_stride", 5)]
        teach = S["FB_ANNOT"][200:]; pool = S["FB_ANNOT"][:200]
    else:
        base_outs = json.load(open(f"/r/l9b/res/base_{pre}_EXAM.json")); nid = set(json.load(open(f"/root/l9b/data/{pre}_neighbours.json")))
        neigh = [(r, o) for r, o in zip(S["EXAM"], base_outs["outs"]) if r["id"] in nid and core.norm(o) == r["answer"].lower()]
        teach = S["FB_TEACH"]; pool = S["FB_ANNOT"]
    err_rows = S["CAL"][:job.get("nerr_pool", 1231)]; eo = R.gen(err_rows); old_err = [(r["ids"], r["answer"]) for r, o in zip(err_rows, eo) if core.norm(o) != r["answer"].lower()]
    log("old-skill errors of packed model on cal pool:", len(old_err), "of", len(err_rows))
    results = []
    for cfg in cfgs:
        TG = cfg.get("target", "shared_down"); install(list(range(NLAY - cfg["nlayers"], NLAY)), TG); LAY = list(range(NLAY - cfg["nlayers"], NLAY)); lam, awt, M, r_col = cfg.get("lam", 1000.0), cfg.get("awt", 10.0), cfg.get("m", 16.0), cfg.get("cols", 512)
        W0 = {l: tmod(l, TG).weight.data.clone() for l in LAY}; HO = W0[LAY[0]].shape[0]
        def restore():
            for l in LAY: tmod(l, TG).weight.data.copy_(W0[l])
        Cs, Ca = cov(LAY, TG); log("covariances done", cfg["name"])
        def edit(items):
            seqs = [ids + tgt_ids(lab) for ids, lab in items]
            def measure():
                Lm = fwd(seqs); x = CAP["fin"].float(); rows = []
                for b, (ids, lab) in enumerate(items):
                    for j, tj in enumerate(tgt_ids(lab)):
                        p = len(ids) - 1 + j; xx = x[b, p]; lg = (G * xx / torch.sqrt((xx ** 2).mean() + eps)) @ WU.T; rows.append((b, p, tj, lg))
                return x, rows, Lm
            x0, rows, Lm = measure(); Sx = []
            for b, p, tj, lg in rows:
                c = int(torch.argmax(lg))
                if c != tj:
                    xx = x0[b, p]; u = G * (WU[tj] - WU[c]); rms = torch.sqrt((xx ** 2).mean() + eps); Sx.append((b, p, xx + rms * (float(lg[c] - lg[tj]) + M) * u / (u @ u)))
            if not Sx: return 0
            for i, l in enumerate(LAY):
                x, _, Lm = measure(); K = torch.stack([keys(l, b, p, Lm) for b, p, _ in Sx]); Rr = torch.stack([tg - x[b, p] for b, p, tg in Sx]) / (len(LAY) - i)
                B = lam * (Cs[l] + awt * Ca[l]); rg = cfg.get("ridge")
                A = B + K.T @ K + (1e-4 * torch.eye(B.shape[0], device=dev) * B.diagonal().mean() if rg is None else lam * rg * Cs[l].diagonal().mean() * torch.eye(B.shape[0], device=dev)); rhs = (Rr.T @ K).T
                dW = torch.linalg.solve(A, rhs).T    # [H, DI]
                if r_col < dW.shape[1]:
                    top = dW.norm(dim=0).topk(r_col).indices.sort().values; As = A[top][:, top]; dWs = torch.linalg.solve(As, rhs[top]).T
                    dW = torch.zeros_like(dW); dW[:, top] = dWs
                tmod(l, TG).weight.data.add_(dW.to(torch.bfloat16))
            return len(Sx)
        def eval_all(tag, full):
            to = R.gen(teach); ta = [core.norm(o) == NEW.lower() for o in to]
            nb = neigh if full else neigh[::3]; no = R.gen([r for r, _ in nb]); ok = [core.norm(o) == r["answer"].lower() for o, (r, _) in zip(no, nb)]
            cells = collections.defaultdict(lambda: [0, 0])
            for o, (r, _) in zip(no, nb): c = cells[r["src"] + "|" + r["kind"]]; c[0] += 1; c[1] += int(core.norm(o) == r["answer"].lower())
            return dict(tag=tag, teach=sum(ta) / len(ta), neigh=sum(ok) / len(ok), neigh_lo=core.wilson(sum(ok), len(ok)), n_neigh=len(ok), min_cell=min(v[1] / v[0] for v in cells.values() if v[0] >= 8),
                        leak=sum(1 for o in no if core.norm(o) == NEW.lower()) / len(no), teach_sample=to[:4])
        out = dict(name=cfg["name"], cfg=cfg, n_old_err=len(old_err), draws=[]); out["empty"] = eval_all("empty", True); log("empty", out["empty"])
        skills = sorted({r["answer"] for r in S["CAL"][:300]})
        for d in range(cfg.get("ndraw", 3)):
            rng = random.Random(seed0 + d); nn_ = cfg.get("n_new", 12); ann_new = rng.sample(pool, nn_); ann_old = rng.sample(old_err, min(len(old_err), 25 - nn_)); items = [(r["ids"] if "ids" in r else R.encode([r])[0]["ids"], NEW) for r in ann_new] + list(ann_old)
            labels = [l for _, l in items]; draw = dict(draw=d, n_ann=len(items))
            for cond in cfg.get("conds", ("real", "shuffled", "wrong")):
                if cond == "real": its = items
                elif cond == "shuffled": l2 = labels[:]; rng.shuffle(l2); its = [(ids, l) for (ids, _), l in zip(items, l2)]
                else: its = [(ids, rng.choice([s for s in skills if s != l])) for ids, l in items]
                ns = edit(its); res = eval_all(cond, cond == "real" or DEV); res["n_sites"] = ns
                if cond == "real" and d == 0:   # carrier bytes: r replaced fp16 columns (+int16 idx) per edited layer
                    carrier = 0
                    for l in LAY:
                        dWl = tmod(l, TG).weight.data.float() - W0[l].float(); nz = int((dWl.abs().sum(0) > 0).sum()); carrier += nz * (HO * 2 + 2)
                    res["carrier_bytes"] = carrier
                draw[cond] = res; restore(); log("draw", d, cond, {k: v for k, v in res.items() if k != "teach_sample"})
            out["draws"].append(draw)
        results.append(out); json.dump(results, open(f"/r/l9b/res/fb_{job['art']}_{job.get('tag','fb')}.json", "w")); commit()
    print("RESULT_JSON", json.dumps(results), flush=True)
