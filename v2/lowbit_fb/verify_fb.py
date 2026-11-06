"""On-box verification with the REAL artifact code path (DEV/POOL only).
python verify_fb.py ART DEVJSONL POOLJSONL OUTJSON
 1. dev acc: head OFF (TENPCT_NOHEAD semantics) vs head ON, per task; 2. ingest k errors (pool) for a task -> memory file -> dev with memory,
 bytes of memory; 3. new-label ingest (dbpedia Film) k annotations -> new recall / old."""
import sys, json, os, time, numpy as np, importlib.util, collections
art, devf, poolf, outf = sys.argv[1:5]
spec = importlib.util.spec_from_file_location("predict", f"{art}/predict.py"); P = importlib.util.module_from_spec(spec); spec.loader.exec_module(P)
f = P.load(art); B = f.build; H = P.Head(art); L = "ABCDEFGHIJKLMN"
dev = [json.loads(x) for x in open(devf)]; pool = [json.loads(x) for x in open(poolf)]
view = lambda rs: [{k: v for k, v in r.items() if k != "answer"} for r in rs]
y = np.array([r["answer"] for r in dev]); res = {}
lg, FT = f.base(view(dev))                                  # biased base logits + feats, computed once
def dec(rows, LG, FT_, M, Hh): return np.array([P._decide(Hh, M, r, LG[i], FT_[i]) for i, r in enumerate(rows)])
M0 = P.Mem(); pb = np.array([int(l[:len(r["names"])].argmax()) for l, r in zip(lg, dev)]); ph = dec(dev, lg, FT, M0, H)
res["dev_base_acc"] = float((pb == y).mean()); res["dev_head_acc"] = float((ph == y).mean()); res["raw_acc"] = float((f.logits(view(dev)).argmax(1) == y).mean())
res["per_task"] = {t: dict(base=float((pb == y)[[r["task"] == t for r in dev]].mean()), head=float((ph == y)[[r["task"] == t for r in dev]].mean())) for t in H.T}
print("DEV", json.dumps(res), flush=True)
# ---- scenario B on real code path
tmp = "/tmp/fbmem"; os.makedirs(tmp, exist_ok=True); byk = {}
yp = np.array([r["answer"] for r in pool]); lgp, FTp = f.base(view(pool)); php = dec(pool, lgp, FTp, M0, H)
for tn in ("ag_news", "dbpedia", "offensive", "yelp"):
    mt = np.array([r["task"] == tn for r in dev]); wrong = [i for i, r in enumerate(pool) if r["task"] == tn and php[i] != yp[i]]
    for k in (1, 5, 10, 25):
        accs = []
        for sd in range(3):
            sel = list(np.random.RandomState(sd).permutation(wrong)[:k])
            if len(sel) < k: continue
            ann = [dict(pool[i], label=pool[i]["names"][pool[i]["answer"]]) for i in sel]
            mp = f"{tmp}/m_{tn}_{k}_{sd}.json"; nb = P.ingest(art, ann, None, mp, B=B)
            M = P.Mem(mp); pd = dec(dev, lg, FT, M, H); pa = dec([pool[i] for i in sel], lgp[sel], FTp[sel], M, H)
            wr = mt & (ph != y)
            accs.append(dict(bytes=nb, task_dev_acc=float((pd == y)[mt].mean()), fix=float((pd == y)[wr].mean()) if wr.sum() else None, corr_ok=float((pd == y)[mt & (ph == y)].mean()),
                             total_dev_acc=float((pd == y).mean()), other_tasks_unchanged=bool((pd[~mt] == ph[~mt]).all()), rec=float((pa == yp[sel]).mean())))
        byk[f"B_{tn}_{k}"] = accs
        print("B", tn, k, accs, flush=True)
res["B"] = byk
json.dump(res, open(outf, "w"), indent=1); print("VERIFY_DONE")
