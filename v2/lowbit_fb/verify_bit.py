"""Real-code-path check of the BitNet-2B artifact_code/predict.py (DEV/POOL). python verify_bit.py ART dev pool out"""
import sys, json, os, numpy as np, importlib.util
art, devf, poolf, outf = sys.argv[1:5]
spec = importlib.util.spec_from_file_location("predict", f"{art}/predict.py"); P = importlib.util.module_from_spec(spec); spec.loader.exec_module(P)
dev = [json.loads(x) for x in open(devf)]; pool = [json.loads(x) for x in open(poolf)]; view = lambda rs: [{k: v for k, v in r.items() if k != "answer"} for r in rs]
y = np.array(["ABCDEFGHIJKLMN"[r["answer"]] for r in dev]); res = {}
f = P.load(art); pd = np.array(f(view(dev))); res["dev_acc_head"] = float((pd == y).mean())
os.environ["TENPCT_NOHEAD"] = "1"; f0 = P.load(art); p0 = np.array(f0(view(dev))); res["dev_acc_nohead"] = float((p0 == y).mean()); print(res, flush=True)
B = f.__closure__ and None
import gc
# ingest through the real function (reuses a fresh build)
os.environ.pop("TENPCT_NOHEAD"); Bd = P._build(art); H = P.Head(art); M0 = P.Mem()
yp = [r["answer"] for r in pool]; tn = "dbpedia"
wrong = [i for i, r in enumerate(pool) if r["task"] == tn and "ABCDEFGHIJKLMN"[P._decide(H, M0, r, *P._fwd(Bd, r))] != "ABCDEFGHIJKLMN"[r["answer"]]]
mt = np.array([r["task"] == tn for r in dev]); out = {}
for k in (1, 5, 10, 25):
    sel = list(np.random.RandomState(0).permutation(wrong)[:k]); ann = [dict(pool[i], label=pool[i]["names"][pool[i]["answer"]]) for i in sel]
    mp = f"/tmp/mb_{k}.json"; nb = P.ingest(art, view(ann) and [dict(a, label=pool[sel[j]]["names"][pool[sel[j]]["answer"]]) for j, a in enumerate(view(ann))], None, mp, B=Bd)
    fm = P.load(art, mp) if False else None
    out[k] = dict(bytes=nb)
    lgs = []
    # evaluate dev with memory using decide on cached forward
    if k == 1:
        cache = [P._fwd(Bd, r) for r in view(dev)]
    M = P.Mem(mp); p = np.array(["ABCDEFGHIJKLMN"[P._decide(H, M, r, cache[i][0], cache[i][1])] for i, r in enumerate(view(dev))])
    out[k].update(task_acc=float((p == y)[mt].mean()), total=float((p == y).mean()), others_unchanged=bool((p[~mt] == pd[~mt]).all())); print(k, out[k], flush=True)
res["ingest_dbpedia"] = out; json.dump(res, open(outf, "w"), indent=1); print("VERIFY_DONE")
