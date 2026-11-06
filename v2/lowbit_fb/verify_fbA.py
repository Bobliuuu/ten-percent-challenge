"""Real-code-path new-label test (DEV/POOL only): hold out dbpedia label 12 (shipped head still contains it in PCA; its class is NOT used: the row names omit it
and a fresh class name 'NEWLABEL' is introduced through ingest). python verify_fbA.py ART dev pool out"""
import sys, json, os, numpy as np, importlib.util
art, devf, poolf, outf = sys.argv[1:5]
spec = importlib.util.spec_from_file_location("predict", f"{art}/predict.py"); P = importlib.util.module_from_spec(spec); spec.loader.exec_module(P)
f = P.load(art); B = f.build; H = P.Head(art); OTHER = "Other / none of these"; h = 12
def hold(rs):
    out = []
    for r in rs:
        if r["task"] != "dbpedia": continue
        idx = [i for i in range(len(r["names"])) if i != h]; rr = dict(r); rr["names"] = [r["names"][i] for i in idx] + [OTHER]; rr["ytrue"] = r["answer"]
        rr["answer"] = idx.index(r["answer"]) if r["answer"] != h else len(idx); out.append(rr)
    return out
dev = hold([json.loads(x) for x in open(devf)]); pool = hold([json.loads(x) for x in open(poolf)])
view = lambda rs: [{k: v for k, v in r.items() if k not in ("answer", "ytrue")} for r in rs]
lg, FT = f.base(view(dev)); y = np.array([r["answer"] for r in dev]); new = np.array([r["ytrue"] == h for r in dev])
def dec(M): return np.array([P._decide(H, M, r, lg[i], FT[i]) for i, r in enumerate(dev)])
nold = len(dev[0]["names"]) - 1; p0 = dec(P.Mem()); res = dict(base=dict(new=float((p0[new] == nold).mean()), old=float((p0[~new] == y[~new]).mean())))
pn = [r for r in pool if r["ytrue"] == h]
for k in (1, 5, 10, 25):
    rs = []
    for sd in range(5):
        sel = list(np.random.RandomState(sd).permutation(len(pn))[:k]); ann = [dict(pn[i], label="NEWLABEL") for i in sel]
        mp = f"/tmp/mA_{k}_{sd}.json"; nb = P.ingest(art, view(ann) and [dict(a, label="NEWLABEL") for a in view(ann)], None, mp, B=B)
        p = dec(P.Mem(mp)); rs.append(dict(bytes=nb, new=float((p[new] == nold).mean()), old=float((p[~new] == y[~new]).mean()), fa=float((p[~new] == nold).mean())))
    res[k] = {q: float(np.mean([r[q] for r in rs])) for q in rs[0]}; print("A", k, res[k], flush=True)
print("base", res["base"]); json.dump(res, open(outf, "w"), indent=1); print("VERIFY_DONE")
