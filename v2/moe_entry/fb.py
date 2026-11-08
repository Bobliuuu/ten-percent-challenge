"""Diffuse closed-form feedback edit on o_proj (attention output projection) of the last layers of the compressed MoE; real vs shuffled vs wrong vs empty.
Mechanism (no optimiser, no new tensor): teacher-force each annotated (input, label); at answer positions where argmax != label token take the final residual x and the closed-form residual target
dx = rms(x) (gap+m) u/(u.u), u = gamma (W_U[t]-W_U[c]); ascending over layers, layer l delivers 1/(layers left) of the CURRENT remaining dx via o_proj: dW = R^T K (lam (C + awt C_ans + ridge I) + K^T K)^-1."""
import torch, json, random, collections, time, math

def wilson(k, n, z=1.96):
    if n == 0: return 0.0
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); r = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)); return (c - r) / d

@torch.no_grad()
def feedback(cfg, X):
    lm, tok, dev, chat, gen_eval, getset, log = X["lm"], X["tok"], "cuda", X["chat"], X["gen_eval"], X["getset"], X["log"]
    NEW = cfg.get("new", "calculator"); LAY = cfg["fb_layers"]; lam = cfg.get("lam", 1000.0); awt = cfg.get("awt", 10.0); M = cfg.get("m", 16.0); ndraw = cfg.get("ndraw", 4); NA = 24
    mod = cfg.get("fb_mod", "o_proj")
    layers = lm.model.layers; EOS = "<|im_end|>"
    getp = lambda l: getattr(layers[l].self_attn, "o_proj").weight
    backup = {l: getp(l).data.clone() for l in LAY}
    restore = lambda: [getp(l).data.copy_(backup[l]) for l in LAY]
    cap = {}; hooks = []
    for l in LAY: hooks.append(layers[l].self_attn.o_proj.register_forward_pre_hook(lambda m_, a, l=l: cap.__setitem__(l, a[0].detach())))
    hooks.append(lm.model.norm.register_forward_pre_hook(lambda m_, a: cap.__setitem__("fin", a[0].detach())))
    WU = lm.lm_head.weight.float(); G = lm.model.norm.weight.float(); eps = lm.config.rms_norm_eps
    tgt_ids = lambda lab: tok(lab + EOS, add_special_tokens=False).input_ids
    def fwd(seqs):
        L = max(len(s) for s in seqs); ids = torch.full((len(seqs), L), tok.pad_token_id); am = torch.zeros((len(seqs), L), dtype=torch.long)
        for r, s in enumerate(seqs): ids[r, :len(s)] = torch.tensor(s); am[r, :len(s)] = 1
        lm.model(input_ids=ids.to(dev), attention_mask=am.to(dev), use_cache=False)
    # ---- covariance over calibration prompts + gold answers
    cal = [r for r in getset("cal")[:cfg.get("ncovs", 300)]]
    Cs = {l: torch.zeros(4096, 4096, device=dev) for l in LAY}; Ca = {l: torch.zeros(4096, 4096, device=dev) for l in LAY}; nt = 0; na = 0
    for s0 in range(0, len(cal), 16):
        bt = cal[s0:s0 + 16]; seqs = [r["ids"] + tgt_ids(r["answer"]) for r in bt]; fwd(seqs)
        for l in LAY:
            for b, r in enumerate(bt):
                k = cap[l][b, :len(seqs[b])].float(); Cs[l] += k.T @ k; ka = k[len(r["ids"]) - 1:]; Ca[l] += ka.T @ ka
        nt += sum(len(s) for s in seqs); na += sum(len(tgt_ids(r["answer"])) for r in bt)
    for l in LAY: Cs[l] /= nt; Ca[l] /= na
    log("fb covariances done")
    def edit(items):
        """items: list of (prompt_ids, label). returns number of sites"""
        seqs, sites = [], []
        for ids, lab in items:
            t = tgt_ids(lab); seqs.append(ids + t)
        def measure():
            fwd(seqs); x = cap["fin"].float(); rows = []
            for b, (ids, lab) in enumerate(items):
                t = tgt_ids(lab)
                for j, tj in enumerate(t):
                    p = len(ids) - 1 + j; xx = x[b, p]; lg = (G * xx / torch.sqrt((xx ** 2).mean() + eps)) @ WU.T
                    rows.append((b, p, tj, lg))
            return x, rows
        x0, rows = measure(); S = []
        for b, p, tj, lg in rows:
            c = int(torch.argmax(lg)); 
            if c != tj:
                xx = x0[b, p]; u = G * (WU[tj] - WU[c]); rms = torch.sqrt((xx ** 2).mean() + eps); dx = rms * (float(lg[c] - lg[tj]) + M) * u / (u @ u); S.append((b, p, xx + dx))
        if not S: return 0
        for i, l in enumerate(LAY):
            x, _ = measure(); K = torch.stack([cap[l][b, p].float() for b, p, _ in S]); R = torch.stack([tg - x[b, p] for b, p, tg in S]) / (len(LAY) - i)
            A = lam * (Cs[l] + awt * Ca[l]) + K.T @ K + 1e-4 * torch.eye(4096, device=dev) * (lam * (Cs[l] + awt * Ca[l])).diagonal().mean()
            dW = torch.linalg.solve(A, (R.T @ K).T).T   # [2048,4096]
            getp(l).data.add_(dW.to(getp(l).dtype))
        return len(S)
    # ---- data
    DSN = cfg["ds"]; calc_pool = [r for r in getset(cfg.get("ann_set", "fb_annot24"))]; teach = getset(cfg.get("teach_set", "fb_teach_unseen"))
    NS = cfg.get("neigh_sets", ["SEL", "REP"]); base = {sn: json.load(open(f"/r/res/base_{DSN}_{sn}_1.json")) for sn in NS}
    neigh = [(r, o) for sn in NS for r, o in zip(getset(sn), base[sn]["outs"]) if o == r["answer"]]
    errs_rows = [r for r in getset("cal")[:cfg.get("nerr_pool", 1500)]]; eo = gen_eval(errs_rows); old_err = [(r["ids"], r["answer"]) for r, o in zip(errs_rows, eo) if o != r["answer"]]
    log("old-skill errors of compressed model on cal pool:", len(old_err), "of", len(errs_rows))
    out = dict(name=cfg["name"] + "_fb", cfg=cfg, n_old_err=len(old_err), draws=[])
    def eval_all(tag, full=True):
        to = gen_eval(teach); ta = [o == NEW for o in to]; srcs = sorted({r["src"] for r in teach}); by_src = {sc: sum(ta[i] for i, r in enumerate(teach) if r["src"] == sc) / sum(1 for r in teach if r["src"] == sc) for sc in srcs}
        nb = neigh if full else neigh[::3]; no = gen_eval([r for r, _ in nb]); ok = [o == r["answer"] for o, (r, _) in zip(no, nb)]
        cells = collections.defaultdict(lambda: [0, 0])
        for o, (r, _) in zip(no, nb): c = cells[r["src"] + "|" + r["kind"]]; c[0] += 1; c[1] += int(o == r["answer"])
        leak = sum(1 for o in no if o == NEW) / len(no); extra = {}
        if cfg.get("neigh_ids_file"):
            NID = set(json.load(open(cfg["neigh_ids_file"]))); sub = [o == r["answer"] for o, (r, _) in zip(no, nb) if r["id"] in NID]; extra = dict(neigh_sub=sum(sub) / max(1, len(sub)), n_sub=len(sub), neigh_sub_lo=wilson(sum(sub), len(sub)))
        d = dict(tag=tag, teach=sum(ta) / len(ta), teach_by_src=by_src, neigh=sum(ok) / len(ok), neigh_lo=wilson(sum(ok), len(ok)), n_neigh=len(ok),
                 min_cell=min(v[1] / v[0] for v in cells.values()), **extra, cells={k: round(v[1] / v[0], 4) for k, v in sorted(cells.items())}, leak=leak)
        return d
    out["empty"] = eval_all("empty"); log("empty", {k: v for k, v in out["empty"].items() if k != "cells"})
    for d in range(ndraw):
        rng = random.Random(100 + d); ann_new = rng.sample(calc_pool, cfg.get("n_new", 12)); ann_old = rng.sample(old_err, min(len(old_err), NA - cfg.get("n_new", 12)))
        items = [(r["ids"], NEW) for r in ann_new] + list(ann_old); labels = [l for _, l in items]
        draw = dict(draw=d, n_ann=len(items))
        for cond in cfg.get("conds", ["real", "shuffled", "wrong"]):
            if cond == "real": its = items
            elif cond == "shuffled":   # strict derangement across groups: every taught-class input gets an OLD label and the old-skill inputs get the NEW label
                nn_ = len(ann_new); olds = [l for _, l in ann_old]; its = [(items[i][0], olds[i % len(olds)]) for i in range(nn_)] + [(items[nn_ + j][0], NEW) for j in range(len(ann_old))]
            elif cond == "wrong":
                skills = sorted({r["answer"] for r in getset("cal")[:300]}); its = [(ids, rng.choice([s for s in skills if s != l])) for ids, l in items]
            ns = edit(its); res = eval_all(cond, full=(cond == "real")); res["n_sites"] = ns; draw[cond] = res; restore()
            log("draw", d, cond, {k: v for k, v in res.items() if k != "cells"})
        out["draws"].append(draw)
    for h in hooks: h.remove()
    print("RESULT_JSON", json.dumps(out), flush=True); return out
