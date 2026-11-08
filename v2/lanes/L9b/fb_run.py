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
    def mk_pre(l):
        def f(m, a):
            if CAP["on"]: CAP[("k", l)] = a[0].detach()
        return f
    def mk_gate(l):
        def f(m, a, o):
            if CAP["on"]: CAP[("g", l)] = torch.sigmoid(o.detach().float())
        return f
    for l in range(NLAY):
        mlp = layers[l].mlp
        hooks.append(mlp.shared_expert.down_proj.register_forward_pre_hook(mk_pre(l))); hooks.append(mlp.shared_expert_gate.register_forward_hook(mk_gate(l)))
    hooks.append(model.model.norm.register_forward_pre_hook(lambda m, a: CAP.__setitem__("fin", a[0].detach()) if CAP["on"] else None))
    def fwd(seqs):
        torch.cuda.empty_cache(); Lm = max(len(s) for s in seqs); ids = torch.full((len(seqs), Lm), tok.pad_token_id); am = torch.zeros((len(seqs), Lm), dtype=torch.long)
        for r, s in enumerate(seqs): ids[r, :len(s)] = torch.tensor(s); am[r, :len(s)] = 1
        CAP["on"] = True; model.model(input_ids=ids.to(dev), attention_mask=am.to(dev), use_cache=False); CAP["on"] = False; return Lm
    def keys(l, b, p, Lm): return (CAP[("k", l)][b * Lm + p].float() * CAP[("g", l)][b * Lm + p].float()).to(dev)
    cal = S["CAL"][:job.get("ncovs", 300)]; R.encode(cal)
    def cov(LAY):
        DI = layers[LAY[0]].mlp.shared_expert.down_proj.weight.shape[1]
        Cs = {l: torch.zeros(DI, DI, device=dev) for l in LAY}; Ca = {l: torch.zeros(DI, DI, device=dev) for l in LAY}; nt = na = 0
        for s0 in range(0, len(cal), 16):
            bt = cal[s0:s0 + 16]; seqs = [r["ids"] + tgt_ids(r["answer"]) for r in bt]; Lm = fwd(seqs)
            for l in LAY:
                for b, r in enumerate(bt):
                    k = (CAP[("k", l)][b * Lm:b * Lm + len(seqs[b])].float() * CAP[("g", l)][b * Lm:b * Lm + len(seqs[b])].float()); Cs[l] += k.T @ k; ka = k[len(r["ids"]) - 1:]; Ca[l] += ka.T @ ka
            nt += sum(len(s) for s in seqs); na += sum(len(tgt_ids(r["answer"])) for r in bt)
        for l in LAY: Cs[l] /= nt; Ca[l] /= na
        return Cs, Ca
    base_outs = {n: json.load(open(f"/r/l9b/res/base_skill7_{n}.json")) for n in ("SEL", "REP")}
    neigh = [(r, o) for n in ("SEL", "REP") for r, o in zip(S[n], base_outs[n]["outs"]) if core.norm(o) == r["answer"].lower()] if "SEL" in S and "REP" in S else []
    teach = S["FB_TEACH"]; pool = S["FB_ANNOT"]; NEW = "Home"
    err_rows = S["CAL"][:job.get("nerr_pool", 1231)]; eo = R.gen(err_rows); old_err = [(r["ids"], r["answer"]) for r, o in zip(err_rows, eo) if core.norm(o) != r["answer"].lower()]
    log("old-skill errors of packed model on cal pool:", len(old_err), "of", len(err_rows))
    results = []
    for cfg in cfgs:
        LAY = list(range(NLAY - cfg["nlayers"], NLAY)); lam, awt, M, r_col = cfg.get("lam", 1000.0), cfg.get("awt", 10.0), cfg.get("m", 16.0), cfg.get("cols", 512)
        W0 = {l: layers[l].mlp.shared_expert.down_proj.weight.data.clone() for l in LAY}
        def restore():
            for l in LAY: layers[l].mlp.shared_expert.down_proj.weight.data.copy_(W0[l])
        Cs, Ca = cov(LAY); log("covariances done", cfg["name"])
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
                B = lam * (Cs[l] + awt * Ca[l]); A = B + K.T @ K + 1e-4 * torch.eye(B.shape[0], device=dev) * B.diagonal().mean(); rhs = (Rr.T @ K).T
                dW = torch.linalg.solve(A, rhs).T    # [H, DI]
                if r_col < dW.shape[1]:
                    top = dW.norm(dim=0).topk(r_col).indices.sort().values; As = A[top][:, top]; dWs = torch.linalg.solve(As, rhs[top]).T
                    dW = torch.zeros_like(dW); dW[:, top] = dWs
                layers[l].mlp.shared_expert.down_proj.weight.data.add_(dW.to(torch.bfloat16))
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
            rng = random.Random(100 + d); nn_ = cfg.get("n_new", 12); ann_new = rng.sample(pool, nn_); ann_old = rng.sample(old_err, min(len(old_err), 25 - nn_)); items = [(r["ids"] if "ids" in r else R.encode([r])[0]["ids"], NEW) for r in ann_new] + list(ann_old)
            labels = [l for _, l in items]; draw = dict(draw=d, n_ann=len(items))
            for cond in ("real", "shuffled", "wrong"):
                if cond == "real": its = items
                elif cond == "shuffled": l2 = labels[:]; rng.shuffle(l2); its = [(ids, l) for (ids, _), l in zip(items, l2)]
                else: its = [(ids, rng.choice([s for s in skills if s != l])) for ids, l in items]
                ns = edit(its); res = eval_all(cond, cond == "real"); res["n_sites"] = ns
                if cond == "real" and d == 0:   # carrier bytes: r replaced fp16 columns (+int16 idx) per edited layer
                    carrier = 0
                    for l in LAY:
                        dWl = layers[l].mlp.shared_expert.down_proj.weight.data.float() - W0[l].float(); nz = int((dWl.abs().sum(0) > 0).sum()); carrier += nz * (H * 2 + 2)
                    res["carrier_bytes"] = carrier
                draw[cond] = res; restore(); log("draw", d, cond, {k: v for k, v in res.items() if k != "teach_sample"})
            out["draws"].append(draw)
        results.append(out); json.dump(results, open(f"/r/l9b/res/fb_{job['art']}_{job.get('tag','fb')}.json", "w")); commit()
    print("RESULT_JSON", json.dumps(results), flush=True)
