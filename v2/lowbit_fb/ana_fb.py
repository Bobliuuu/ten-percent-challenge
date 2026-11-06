"""lowbit_fb analysis (numpy; features from feat_fb.py, DECODED BitNet-2B artifact). Closed-form head + count updates only.
python ana_fb.py <featdir> [export]   -> results_fb.json (+ head params), prints tables; `export` writes artifact_code/head.npz"""
import numpy as np, json, sys, itertools, os
D = sys.argv[1]; SEEDS = 10; KS = (1, 5, 10, 25); R = 96; ALPHA = float(os.environ.get("ALPHA", "0.1"))
TASKS = ["ag_news", "dbpedia", "yelp", "amazon", "emotion", "offensive"]; HOLD = {"ag_news": 3, "dbpedia": 12, "emotion": 2}
COLSN = os.environ.get("COLS", "both"); OUT = os.environ.get("OUT", "results_fb.json"); HEADOUT = os.environ.get("HEADOUT", "artifact_code/head.npz")
FULL = [None]
def ld(n):
    z = np.load(f"{D}/{n}.npz"); d = {k: z[k] for k in z.files}; F = d["X"].shape[1]; FULL[0] = F; h = F // 2
    d["X"] = d["X"][:, {"both": slice(0, F), "last": slice(0, h), "mean": slice(h, F)}[COLSN]]; return d
def lsm(L): L = L - L.max(1, keepdims=True); return L - np.log(np.exp(L).sum(1, keepdims=True))
class Head:
    """PCA(R) over all calib features; shared shrunk within-class inverse covariance pooled over all (task,class) groups."""
    def __init__(s, X, grp):
        X = X.astype(np.float64); s.mu0 = X.mean(0); Z = X - s.mu0
        s.P = np.linalg.svd(Z, full_matrices=False)[2][:R]; Zp = Z @ s.P.T; W = Zp.copy()
        for g in np.unique(grp): W[grp == g] -= Zp[grp == g].mean(0)
        C = W.T @ W / len(W); s.Ci = np.linalg.inv((1 - ALPHA) * C + ALPHA * np.trace(C) / R * np.eye(R))
    def proj(s, x): return (x.astype(np.float64) - s.mu0) @ s.P.T
    def score(s, z, mus): M = np.asarray(mus); return z @ (s.Ci @ M.T) - 0.5 * np.einsum("cr,rs,cs->c", M, s.Ci, M)[None, :]
def upd(mus, cnt, Zs, ys, w):
    mus = [m.copy() for m in mus]; cnt = list(cnt)
    for z, c in zip(Zs, ys): mus[c] = (cnt[c] * mus[c] + w * z) / (cnt[c] + w); cnt[c] += w
    return mus, cnt
def agg(rs): return {q: float(np.nanmean([r[q] for r in rs])) for q in rs[0]}
def sub(z, m): return {k: v[m] for k, v in z.items()}
def setup():
    cal, pool, dev = ld("mdc_calib"), ld("mdc_pool"), ld("mdc_dev")
    grp = np.array([f"{t}|{y}" for t, y in zip(cal["task"], cal["y"])]); H = Head(cal["X"], grp)
    Zc = H.proj(cal["X"]); Zp = H.proj(pool["X"]); Zd = H.proj(dev["X"]); return cal, pool, dev, H, Zc, Zp, Zd
LAMS = (0.03, 0.1, 0.3, 1, 3, 10)
def fit_task(tn, cal, H, Zc):
    m = cal["task"] == tn; y = cal["y"][m]; n = int(y.max()) + 1; Z = Zc[m]
    return [Z[y == c].mean(0) for c in range(n)], [float((y == c).sum()) for c in range(n)], n
def k0(tn, S, cal, pool, dev, H, Zc, Zp, Zd):
    """tune lam0 on POOL; report dev with/without. returns lam0 (0 = head OFF for this task)"""
    mus, cnt, n = fit_task(tn, cal, H, Zc)
    def acc(sp, Z, lam):
        m = sp["task"] == tn; L = lsm(sp["L"][m][:, :n]); Sx = H.score(Z[m], mus); return float(((L + lam * Sx).argmax(1) == sp["y"][m]).mean())
    b = acc(pool, Zp, 0); best = max(LAMS, key=lambda l: acc(pool, Zp, l))
    lam0 = best if acc(pool, Zp, best) > b + 0.005 else 0.0
    return dict(lam0=lam0, pool_base=b, pool_head=acc(pool, Zp, best), pool_best_lam=best, dev_base=acc(dev, Zd, 0), dev_head_at_lam0=acc(dev, Zd, lam0), dev_head_at_best=acc(dev, Zd, best))
def scenB(tn, P, split, cal, pool, dev, H, Zc, Zp, Zd):
    mus0, cnt0, n = fit_task(tn, cal, H, Zc); mp = pool["task"] == tn; md = dev["task"] == tn
    Lp = lsm(pool["L"][mp][:, :n]); yp = pool["y"][mp]; Zpp = Zp[mp]
    ev = dev if split == "dev" else pool; me = md if split == "dev" else mp
    Le = lsm(ev["L"][me][:, :n]); ye = ev["y"][me]; Ze = (Zd if split == "dev" else Zp)[me]
    S0 = H.score(Ze, mus0); ship = (Le + P["lam0"] * S0).argmax(1); wrongE = ship != ye
    Sp0 = H.score(Zpp, mus0); wrongp = np.where((Lp + P["lam0"] * Sp0).argmax(1) != yp)[0]
    out = {}
    for k in (0,) + KS:
        rows = {m: [] for m in ("bias", "head_fb", "combo")}
        for sd in range(SEEDS if k else 1):
            sel = np.random.RandomState(sd).permutation(wrongp)[:k]
            if len(sel) < k: continue
            mask = np.ones(len(ye), bool)
            def fin(pred, Lpa, Sa):  # Lpa/Sa: scores for annotated items
                m1 = mask & wrongE; m2 = mask & ~wrongE
                return dict(fix=float((pred[m1] == ye[m1]).mean()) if m1.sum() else np.nan, acc=float((pred[mask] == ye[mask]).mean()),
                            corr_ok=float((pred[m2] == ye[m2]).mean()), rec=float((Lpa.argmax(1) == yp[sel]).mean()) if k else np.nan)
            ya = yp[sel]; pa = (Lp[sel] + P["lam0"] * Sp0[sel]).argmax(1) if k else None
            if k: tt = np.bincount(ya, minlength=n) + P["ba"]; pp = np.bincount(pa, minlength=n) + P["ba"]; b = P["eta"] * np.log(tt / pp)
            else: b = np.zeros(n)
            mus, cnt = upd(mus0, cnt0, Zpp[sel], ya, P["w"]) if k else (mus0, cnt0)
            S = H.score(Ze, mus); Sa = H.score(Zpp[sel], mus) if k else None
            lam = P["lam_fb"] if k else P["lam0"]
            Lpa = Lp[sel] if k else None
            rows["bias"].append(fin((Le + b + P["lam0"] * S0).argmax(1), (Lp[sel] + b + P["lam0"] * Sp0[sel]) if k else None, None))
            rows["head_fb"].append(fin((Le + lam * S).argmax(1), (Lp[sel] + lam * Sa) if k else None, None))
            rows["combo"].append(fin((Le + lam * S + b).argmax(1), (Lp[sel] + lam * Sa + b) if k else None, None))
        out[k] = {m: agg(r) for m, r in rows.items() if r}
    return dict(base_acc=float((Le.argmax(1) == ye).mean()), ship_k0_acc=float((ship == ye).mean())), out
def tuneB(tn, lam0, *a):
    best = None
    for w, lam_fb, eta in itertools.product((3, 10, 30), (0.3, 1, 3), (0.5, 1, 2)):
        P = dict(lam0=lam0, w=w, lam_fb=lam_fb, eta=eta, ba=1); _, o = scenB(tn, P, "pool", *a)
        sc = o[10]["combo"]["fix"] + 3 * (o[10]["combo"]["corr_ok"] - 1)
        if best is None or sc > best[0]: best = (sc, P)
    return best[1]
# ---- scenario A (new label), hold-prompt features; head refit on hold-calib minus new class (as feedback2)
def scenA(tn, P, split):
    h = HOLD[tn]; cal, pool, dev = ld(f"hold_{tn}_calib"), ld(f"hold_{tn}_pool"), ld(f"hold_{tn}_dev")
    nold = int(cal["y"].max()); keep = cal["ytrue"] != h
    Hh = Head(cal["X"][keep], cal["y"][keep]); Zc = Hh.proj(cal["X"][keep]); yc = cal["y"][keep]
    mus0 = [Zc[yc == c].mean(0) for c in range(nold)]; cnt0 = [float((yc == c).sum()) for c in range(nold)]
    ev = dev if split == "dev" else pool; Le = lsm(ev["L"][:, :nold + 1]); te = ev["ytrue"]; ye = ev["y"]; Ze = Hh.proj(ev["X"]); Zp = Hh.proj(pool["X"])
    newm = te == h; old = ~newm; pidx = np.where(pool["ytrue"] == h)[0]
    base = Le.argmax(1)
    def fin(pred, pa=None): return dict(new=float((pred[newm] == nold).mean()), old=float((pred[old] == ye[old]).mean()), fa=float((pred[old] == nold).mean()), rec=float(pa) if pa is not None else np.nan)
    out = {}
    for k in KS:
        rows = {m: [] for m in ("gated", "comb", "bias")}
        for sd in range(SEEDS):
            sel = np.random.RandomState(sd).permutation(pidx)[:k]; Za = Zp[sel]
            mnew = (P["n0"] * np.mean(mus0, 0) + P["w"] * Za.sum(0)) / (P["n0"] + P["w"] * k)
            S = Hh.score(Ze, mus0 + [mnew]); Sa = Hh.score(Za, mus0 + [mnew])
            pg = base.copy(); pg[S.argmax(1) == nold] = nold; rows["gated"].append(fin(pg, (Sa.argmax(1) == nold).mean()))
            rows["comb"].append(fin((Le + P["lam"] * S).argmax(1)))
            La = lsm(pool["L"][sel][:, :nold + 1]); b = max(0, (La[:, :nold].max(1) - La[:, nold]).max()); L2 = Le.copy(); L2[:, nold] += b; rows["bias"].append(fin(L2.argmax(1)))
        out[k] = {m: agg(r) for m, r in rows.items()}
    return dict(base=fin(base)), out
def tuneA(tn):
    best = None
    for n0, w in itertools.product((2, 8), (2, 6, 20)):
        P = dict(n0=n0, w=w, lam=0.3); r, o = scenA(tn, P, "pool")
        sc = np.mean([o[k]["gated"]["new"] + o[k]["gated"]["old"] for k in (5, 10)]) - 3 * max(0, r["base"]["old"] - o[5]["gated"]["old"] - 0.01)
        if best is None or sc > best[0]: best = (sc, P)
    return best[1]
if __name__ == "__main__":
    cal, pool, dev, H, Zc, Zp, Zd = setup(); A = (cal, pool, dev, H, Zc, Zp, Zd); RES = dict(k0={}, B={}, A={}, params={})
    ally = dev["y"]; ntot = len(ally)
    for tn in TASKS:
        r = k0(tn, None, *A); RES["k0"][tn] = r; print("k0", tn, {k: round(v, 4) for k, v in r.items()}, flush=True)
    # global dev gate: total dev accuracy with head (per-task lam0) vs without
    def dev_total(use_head):
        c = 0
        for tn in TASKS:
            md = dev["task"] == tn; mus, cnt, n = fit_task(tn, cal, H, Zc); L = lsm(dev["L"][md][:, :n]); lam = RES["k0"][tn]["lam0"] if use_head else 0
            c += ((L + lam * H.score(Zd[md], mus)).argmax(1) == dev["y"][md]).sum()
        return c / ntot
    RES["dev_total_base"], RES["dev_total_head"] = float(dev_total(False)), float(dev_total(True)); print("DEV total base", RES["dev_total_base"], "head", RES["dev_total_head"])
    RES["head_on"] = bool(RES["dev_total_head"] > RES["dev_total_base"] + 0.003)   # strict gate: needs >= +0.3pt on dev
    if not RES["head_on"]:
        for tn in TASKS: RES["k0"][tn]["lam0"] = 0.0
    for tn in TASKS:
        P = tuneB(tn, RES["k0"][tn]["lam0"], *A); RES["params"][tn] = P; b, o = scenB(tn, P, "dev", *A); RES["B"][tn] = dict(params=P, base=b, byk=o)
        print("B", tn, P, b, flush=True)
        for k in (0,) + KS: print("  k", k, {m: {q: round(v, 3) for q, v in d.items()} for m, d in o[k].items()}, flush=True)
    for tn in HOLD:
        P = tuneA(tn); r, o = scenA(tn, P, "dev"); RES["A"][tn] = dict(params=P, base=r["base"], byk=o); print("A", tn, P, r["base"], flush=True)
        for k in KS: print("  k", k, {m: {q: round(v, 3) for q, v in d.items()} for m, d in o[k].items()}, flush=True)
    json.dump(RES, open(OUT, "w"), indent=1)
    if len(sys.argv) > 2 and sys.argv[2] == "export":
        keys, cmu, cn, T = [], [], [], {}
        for tn in TASKS:
            mus, cnt, n = fit_task(tn, cal, H, Zc); names = None
            T[tn] = dict(lam0=RES["k0"][tn]["lam0"], lam_fb=RES["params"][tn]["lam_fb"], w=RES["params"][tn]["w"], eta=RES["params"][tn]["eta"], ba=1, n0=RES["A"].get(tn, {"params": {"n0": 2}})["params"]["n0"], gate_new=True)
            for c in range(n): keys.append((tn, c)); cmu.append(mus[c]); cn.append(cnt[c])
        # class NAMES from calib rows
        rows = [json.loads(l) for l in open("../alt/calib.jsonl")]; nm = {}
        for r in rows: nm[r["task"]] = r["names"]
        meta = dict(keys=[tn + "\x1f" + nm[tn][c] for tn, c in keys], tasks=T, R=R, alpha=ALPHA)
        F = FULL[0]; cols = {"both": [0, F], "last": [0, F // 2], "mean": [F // 2, F]}[COLSN]; meta["cols"] = cols
        np.savez(HEADOUT, mu0=H.mu0.astype(np.float16), P=H.P.astype(np.float16), Ci=H.Ci.astype(np.float32), cmu=np.array(cmu, np.float32), cn=np.array(cn, np.float32), meta=np.frombuffer(json.dumps(meta).encode(), np.uint8))
        print("head.npz bytes", os.path.getsize(HEADOUT))
