"""Closed-form feedback head for the C2 7B artifact. Local numpy only, features from Volume tenpct-c2fb:/f (pull to feats/).
python ana.py build   -> head.npz (+ tuning log);  python ana.py eval -> results.json"""
import sys, os, json, importlib.util, collections, numpy as np, itertools
D = os.path.dirname(os.path.abspath(__file__)); R = D + "/../commercial/C2/"
spec = importlib.util.spec_from_file_location("predict", D + "/artifact_code/predict.py"); P = importlib.util.module_from_spec(spec); spec.loader.exec_module(P)
def ld(sp):
    rows = [json.loads(l) for l in open(R + f"{sp}.jsonl")]
    for r in rows: r.pop("tools", None); r.pop("gold_calls", None)
    z = np.load(D + f"/feats/{sp}.npz"); return rows, z["L"].astype(np.float32), z["FO"], z["y"]
tn = P.tool_name
def gold(r): return None if r["answer"] == 4 else tn(r["names"][r["answer"]])
def optset(r): return {tn(n) for n in r["names"][:4]}

def build_head(rows, FO, excl, r=64, shr=0.1, params=None):
    keep = [i for i, rw in enumerate(rows) if not (optset(rw) & excl)]
    X = FO[keep].astype(np.float32); y = np.array([rows[i]["answer"] for i in keep])
    mu0 = X[:, :4].reshape(-1, X.shape[-1]).mean(0)
    sub = X.reshape(-1, X.shape[-1])[::3] - mu0
    _, _, Vt = np.linalg.svd(sub, full_matrices=False); Pm = Vt[:r].astype(np.float32)
    U = (X - mu0) @ Pm.T                               # [n,5,r]
    n = len(keep); pos = np.zeros((n, 5), bool); pos[np.arange(n), y] = True
    A = U[:, :4]; pa = pos[:, :4]
    mpos, mneg = A[pa].mean(0), A[~pa].mean(0)
    E = U[:, 4]; pe = pos[:, 4]; mposE, mnegE = E[pe].mean(0), E[~pe].mean(0)
    res = np.concatenate([A[pa] - mpos, A[~pa] - mneg, E[pe] - mposE, E[~pe] - mnegE]); S = res.T @ res / len(res)
    S = (1 - shr) * S + shr * np.trace(S) / r * np.eye(r); Ci = np.linalg.inv(S).astype(np.float32)
    m = np.stack([mpos, mneg, mposE, mnegE]).astype(np.float32)
    return dict(mu0=mu0.astype(np.float32), P=Pm, Ci=Ci, m=m), keep

class H_:   # in-memory Head compatible with predict.Head
    def __init__(s, h, pr): s.mu0, s.P, s.Ci, s.m, s.pr = h["mu0"], h["P"], h["Ci"], h["m"], pr
    proj = P.Head.proj; gscore = P.Head.gscore; tscore = P.Head.tscore
PR0 = dict(lam0=0.0, eta_n=1.0, lam_n=0.0, lam_t=0.5, eta=0.3, ba=1.0, n0t=1.0, w=1.0)

def margins(H, LG, U, G):
    s = P._lsm(LG[:, :5].astype(np.float64)) + H.pr["lam0"] * G; return s[:, 4] - s[:, :4].max(1)

def predict_all(H, M, rows, LG, U, G):
    return np.array([int(P._scores(H, M, LG[i], U[i], G[i], r["names"]).argmax()) for i, r in enumerate(rows)])

# ---------------- vectorised evaluation (verified equal to predict._scores on a sample) ----------------
class Ctx:
    def __init__(s, H, rows, L, FO, y, delta=0.0):
        s.H, s.rows, s.y = H, rows, y; s.L = L.copy(); s.L[:, 4] += delta
        s.U = H.proj(FO); s.G = H.gscore(s.U); s.N = len(rows)
        s.T = np.array([[tn(n) for n in r["names"][:4]] for r in rows])
        s.S0 = P._lsm(s.L[:, :5].astype(np.float64)) + H.pr["lam0"] * s.G
        s.mg = s.S0[:, 4] - s.S0[:, :4].max(1)
    def scores(s, M):
        p = s.H.pr; S = s.S0.copy()
        if M.empty(): return S
        if P.NONE in M.tp: x, z = M.tp[P.NONE]; S[:, 4] += p["eta_n"] * np.log((x + p["ba"]) / (z + p["ba"]))
        if P.NONE in M.proto:
            mu = P._dec(M.proto[P.NONE]["mu"]); d = s.H.Ci @ (mu - s.H.m[3]); S[:, 4] += p["lam_n"] * ((s.U[:, 4] - 0.5 * (mu + s.H.m[3])) @ d)
        for t, e in M.proto.items():
            if t == P.NONE: continue
            m = s.T == t
            if m.any():
                mu = P._dec(e["mu"]); d = s.H.Ci @ (mu - s.H.m[1]); c = 0.5 * (mu + s.H.m[1])
                ii, jj = np.nonzero(m); S[ii, jj] += p["lam_t"] * ((s.U[ii, jj] - c) @ d)
        for t, (a, b) in M.tp.items():
            m = s.T == t; S[:, :4][m] += p["eta"] * np.log((a + p["ba"]) / (b + p["ba"]))
        return S
    def pred(s, M): return s.scores(M).argmax(1)
    def ingest(s, M, idx):
        """sequential, same math as predict.ingest_feats on ctx rows idx"""
        p = s.H.pr
        for i in idx:
            pred = int(s.row_scores(M, i).argmax()); yy = int(s.y[i])
            t = P.NONE if yy == 4 else s.T[i, yy]; e = M.proto.get(t); n, mu = (e["n"], P._dec(e["mu"])) if e else (p["n0t"], s.H.m[2] if yy == 4 else s.H.m[0])
            mu = (n * mu + p["w"] * s.U[i, yy]) / (n + p["w"]); M.proto[t] = {"n": float(n + p["w"]), "mu": P._enc(mu)}
            if yy != 4: M.tp.setdefault(t, [0, 0])[0] += 1
            M.tp.setdefault(P.NONE if pred == 4 else s.T[i, pred], [0, 0])[1] += 1
            if yy == 4: M.tp.setdefault(P.NONE, [0, 0])[0] += 1
        return M
    def row_scores(s, M, i):
        return P._scores(s.H, M, s.L[i], s.U[i], s.G[i], s.rows[i]["names"])

def memsize(M, path="/tmp/_m.json"): M.dump(path); return os.path.getsize(path)
