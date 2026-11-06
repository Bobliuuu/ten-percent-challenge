"""Box version of feat7b.py: python feat_box.py ART PREDICT_PY OUTDIR  (alt splits calib/dev/pool jsonl in cwd; MDC-6 only, never locked)"""
import sys, json, os, time, numpy as np, importlib.util
art, pp, out = sys.argv[1:4]; os.makedirs(out, exist_ok=True)
spec = importlib.util.spec_from_file_location("predict", pp); P = importlib.util.module_from_spec(spec); spec.loader.exec_module(P)
HOLD = {"ag_news": 3, "dbpedia": 12, "emotion": 2}; OTHER = "Other / none of these"; t0 = time.time()
B = P._build(art); print("built", time.time() - t0, flush=True)
ld = lambda sp: [json.loads(l) for l in open(f"{sp}.jsonl")]
def extract(name, rows):
    N = len(rows); lg, X = [], []
    for b in range(0, N, 64):
        l, x = P._base(B, rows[b:b + 64]); lg.append(l); X.append(x.astype(np.float16)); print(name, b, N, time.time() - t0, flush=True)
    LL = np.concatenate(lg); L = np.full((N, 14), -1e4, np.float32); L[:, :LL.shape[1]] = np.where(LL < -1e8, -1e4, LL); X = np.concatenate(X)
    y = np.array([r["answer"] for r in rows]); acc = float((L.argmax(1) == y).mean())
    np.savez(f"{out}/{name}.npz", X=X, L=L, y=y, task=np.array([r["task"] for r in rows]), ytrue=np.array([r.get("ytrue", r["answer"]) for r in rows]))
    print("RESULT_JSON", json.dumps(dict(name=name, n=N, acc=acc, secs=time.time() - t0)), flush=True)
for sp in ("dev", "calib", "pool"): extract("mdc_" + sp, ld(sp))
for sp in ("calib", "dev", "pool"):
    for tn, h in HOLD.items():
        rows = []
        for r in ld(sp):
            if r["task"] != tn: continue
            idx = [i for i in range(len(r["names"])) if i != h]; rr = dict(r); rr["names"] = [r["names"][i] for i in idx] + [OTHER]
            rr["ytrue"] = r["answer"]; rr["answer"] = idx.index(r["answer"]) if r["answer"] != h else len(idx); rows.append(rr)
        extract(f"hold_{tn}_{sp}", rows)
print("ALLDONE")
