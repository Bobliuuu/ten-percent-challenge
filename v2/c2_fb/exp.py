from ana import *
import sys, copy, time
KS = (1, 5, 10, 25); NS = 12
S = {s: ld(s) for s in ("calib", "dev", "pool")}
cr, cL, cFO, cy = S["calib"]; pr_, pL, pFO, py = S["pool"]; dr, dL, dFO, dy = S["dev"]
# annotation sources: calib+pool; tuning sims: annotations from calib + pool-A, eval on pool-B
pid = np.array([int(r["id"][-1], 16) % 2 if isinstance(r.get("id"), str) else i % 2 for i, r in enumerate(pr_)])
LAM0 = 1.0   # chosen on POOL (run.py lam)
def mkctx(H, rows, L, FO, y, delta=0.0): return Ctx(H, rows, L, FO, y, delta)
def prior_margins(H, delta=0.0):   # shipped priors: mean none-margin of none-gold / relevant-gold calib items, UNSHIFTED (shipped state)
    C = Ctx(H, cr, cL, cFO, cy); return float(C.mg[cy == 4].mean()), float(C.mg[cy != 4].mean())
h_all, _ = build_head(cr, cFO, set())
def newH(h, **kw):
    pr = dict(PR0); pr["lam0"] = LAM0; pr.update(kw); return H_(h, pr)

def acc(p, y, m=None): 
    m = np.ones(len(y), bool) if m is None else m
    return float((p[m] == y[m]).mean()) if m.any() else float("nan")

# ------------------------------ S1: none miscalibration ------------------------------
def s1(H, annC, evC, delta, ks=KS, ns=NS, **kw):
    p0 = evC.pred(P.Mem()); err0 = (p0 != evC.y) & ((evC.y == 4) | (p0 == 4)); out = {}
    aerr = np.nonzero((annC.pred(P.Mem()) != annC.y) & ((annC.y == 4) | (annC.pred(P.Mem()) == 4)))[0]
    out["k0"] = dict(unseen=acc(p0, evC.y, err0), dev=acc(p0, evC.y), irr=acc(p0, evC.y, evC.y == 4), rel=acc(p0, evC.y, evC.y != 4), n_err=int(err0.sum()))
    for k in ks:
        r = []
        for sd in range(ns):
            idx = np.random.default_rng(sd).choice(aerr, k, replace=False); M = annC.ingest(P.Mem(), idx); p = evC.pred(M)
            r.append((acc(p, evC.y, err0), acc(p, evC.y), acc(p, evC.y, evC.y == 4), acc(p, evC.y, evC.y != 4), acc(p, evC.y, ~err0), memsize(M)))
        r = np.mean(r, 0); out[f"k{k}"] = dict(unseen=r[0], dev=r[1], irr=r[2], rel=r[3], kept_ok=r[4], bytes=r[5])
    return out

# ------------------------------ S2: new tool ------------------------------
def tool_items(rows, y):
    d = collections.defaultdict(list)
    for i, r in enumerate(rows):
        if y[i] != 4: d[tn(r["names"][y[i]])].append(i)
    return d
def folds(tools, nf=4): return [set(t for j, t in enumerate(sorted(tools)) if j % nf == f) for f in range(nf)]
def s2(annrows_ctx_fn, evCtx_fn, tools, ks, ns, params, nf=4, minpos=None):
    """per fold: head built on calib items without fold tools; per tool: annotations from ann source, eval on eval source"""
    res = {k: [] for k in ks}
    for fs in folds(tools, nf):
        h, _ = build_head(cr, cFO, fs); H = newH(h, **params); annC = annrows_ctx_fn(H); evC = evCtx_fn(H)
        ti_a = tool_items(annC.rows, annC.y); ti_e = tool_items(evC.rows, evC.y)
        p0 = evC.pred(P.Mem()); 
        for t in sorted(fs):
            ia, ie = ti_a.get(t, []), ti_e.get(t, [])
            if len(ie) < 3: continue
            for k in ks:
                if len(ia) < k: continue
                a_, d_, an_, b_, ret0 = [], [], [], [], []
                for sd in range(ns):
                    idx = np.random.default_rng(sd).choice(ia, k, replace=False); M = annC.ingest(P.Mem(), idx); p = evC.pred(M)
                    a_.append((p[ie] == evC.y[ie]).mean()); d_.append((p == evC.y).mean())
                    pa = annC.pred(M); an_.append((pa[idx] == annC.y[idx]).mean()); b_.append(memsize(M))
                res[k].append(dict(t=t, n_ann=len(ia), n_eval=len(ie), base_new=float((p0[ie] == evC.y[ie]).mean()), new=float(np.mean(a_)), base_dev=float((p0 == evC.y).mean()), dev=float(np.mean(d_)), ann_after=float(np.mean(an_)), bytes=float(np.mean(b_))))
    return res
def s2_summ(res):
    o = {}
    for k, v in res.items():
        if not v: continue
        w = np.array([x["n_eval"] for x in v]); f = lambda key: float(np.average([x[key] for x in v], weights=w))
        o[k] = dict(n_tools=len(v), n_eval=int(w.sum()), new_before=f("base_new"), new_after=f("new"), dev_before=float(np.mean([x["base_dev"] for x in v])), dev_after=float(np.mean([x["dev"] for x in v])), ann_after=f("ann_after"), bytes=float(np.mean([x["bytes"] for x in v])))
    return o

# ------------------------------ S3: confusable pair ------------------------------
def s3(H, annC, evC, minann, ks, ns, **kw):
    pa = annC.pred(P.Mem()); pe = evC.pred(P.Mem())
    def pats(C, p):
        d = collections.defaultdict(list)
        for i in np.nonzero((p != C.y) & (C.y != 4) & (p != 4))[0]: d[(C.T[i, C.y[i]], C.T[i, p[i]])].append(i)
        return d
    A, E = pats(annC, pa), pats(evC, pe); res = {k: [] for k in ks}
    for pat, ia in A.items():
        ie = E.get(pat, [])
        if len(ie) < 2: continue
        a, b = pat
        grp = np.array([(a in evC.T[i]) and (b in evC.T[i]) and (evC.y[i] != 4 and evC.T[i, evC.y[i]] in (a, b)) for i in range(evC.N)])  # items offering both tools whose gold is one of them
        for k in ks:
            if len(ia) < k: continue
            r = []
            for sd in range(ns):
                idx = np.random.default_rng(sd).choice(ia, k, replace=False); M = annC.ingest(P.Mem(), idx); p = evC.pred(M)
                r.append(((p[ie] == evC.y[ie]).mean(), (p[grp] == evC.y[grp]).mean() if grp.any() else np.nan, (p == evC.y).mean(), memsize(M)))
            r = np.mean(r, 0); res[k].append(dict(pat=pat, n_ann=len(ia), n_err_eval=len(ie), n_grp=int(grp.sum()), grp_before=float((pe[grp] == evC.y[grp]).mean()) if grp.any() else None, fixed=float(r[0]), grp_after=float(r[1]), dev_before=float((pe == evC.y).mean()), dev=float(r[2]), bytes=float(r[3])))
    return res
def s3_summ(res):
    o = {}
    for k, v in res.items():
        if not v: continue
        w = np.array([x["n_err_eval"] for x in v])
        o[k] = dict(n_pairs=len(v), n_err_eval=int(w.sum()), fixed_unseen_errs=float(np.average([x["fixed"] for x in v], weights=w)), grp_before=float(np.nanmean([x["grp_before"] for x in v])), grp_after=float(np.nanmean([x["grp_after"] for x in v])), dev_before=float(np.mean([x["dev_before"] for x in v])), dev_after=float(np.mean([x["dev"] for x in v])), bytes=float(np.mean([x["bytes"] for x in v])))
    return o
