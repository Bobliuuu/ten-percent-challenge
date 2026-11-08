"""L11 final ranking on the SHARED locked-exam splits (v2/exam): ann_pool (new label), teach (locked unseen), neighbours (exam ids). configs fixed from the ablation BEFORE this run.
usage: python run_exam.py CFG.json OUTDIR   cfg: model, q4, task (SKILL7|BANK6), ndraw, n, conds, methods, cont"""
import json, sys, os, time, random, collections, numpy as np, torch, tempfile
import fblib
from fblib import Env
cfg = json.load(open(sys.argv[1])); OUT = sys.argv[2]; os.makedirs(OUT, exist_ok=True); TASK = cfg["task"]; DD = json.load(open(f"exam_data/{TASK}.json")); NEW = DD["new"]; LABS = DD["labels"]
T0 = time.time(); log = lambda *a: print(f"[{time.time()-T0:6.0f}s]", *a, flush=True)
cal = DD["cal"]; random.Random(0).shuffle(cal)
td = tempfile.mkdtemp(); open(f"{td}/calib_old.jsonl", "w").write("".join(json.dumps(dict(text=r["user"], gold=r["kind"])) + "\n" for r in cal[:cfg.get("ncal", 300)]))
env = Env(cfg["model"], td, q4=cfg.get("q4", False), qbits=cfg.get("qbits", 4), raw=True, ncal=cfg.get("ncal", 300), log=log)
mk = lambda rows: [dict(text=r["user"], gold=r["kind"], src=r["src"], id=r["id"], pid=env.pid(r["user"])) for r in rows]
ann = mk(DD["ann_pool"]); teach = mk(DD["teach"]); neigh = mk(DD["neigh"])
sel = DD["sel"]; random.Random(1).shuffle(sel); cal_eval = mk(sel[:200]); shuf_pool = mk(cal[300:700])
lab = lambda r: NEW; goldf = lambda r: r["gold"]; bits = lambda a: "".join(map(str, a.tolist()))
base_neigh = env.acc(neigh, goldf); NE_OK = np.where(base_neigh == 1)[0].tolist(); gb = env.gen_text([r["pid"] for r in cal_eval], 6); CAL_OK = [i for i, r in enumerate(cal_eval) if gb[i] == r["gold"]]
ANN_BASE = env.gen_text([r["pid"] for r in ann], 6)
base = dict(teach=float(env.acc(teach, lab).mean()), neigh=float(base_neigh.mean()), n_neigh=len(neigh), n_neigh_ok=len(NE_OK), ann_base_labels=dict(collections.Counter(ANN_BASE)))
log("BASE", base)
meta = dict(teach_src=[r["src"] for r in teach], neigh_meta=[(neigh[i]["src"], neigh[i]["gold"]) for i in NE_OK], task=TASK, new=NEW)
cal_acc = lambda: env.acc(cal_eval, goldf, CAL_OK)
def conds(n, d):
    rr = random.Random(100 * n + d); ix = rr.sample(range(len(ann)), n); out = {"real": [(ann[i], NEW) for i in ix]}
    out["wrong"] = [(ann[i], rr.choice([l for l in LABS if l != ANN_BASE[i]])) for i in ix]; sh = rr.sample(range(len(shuf_pool)), n); out["shuffle"] = [(shuf_pool[i], NEW) for i in sh]
    out["_ix"] = ix; return out
RES = dict(cfg=cfg, base=base, meta=meta, runs=[], cont=[]); dump = lambda: json.dump(RES, open(f"{OUT}/res.json", "w"))
prev = None
for mt in cfg["methods"]:
    key = (mt["carrier"], tuple(mt.get("layers", [])))
    if prev != key: env.clear_cache(); prev = key
    env.cache = {k: v for k, v in env.cache.items() if not (isinstance(k, tuple) and k[0] in ("chol", "alphaV"))}; tm = time.time()
    for n in cfg.get("ns", [12]):
        for d in range(cfg["ndraw"]):
            cs = conds(n, d)
            for cn in cfg["conds"]:
                env.restore(); info = env.apply(mt, cs[cn])
                rec = dict(m=mt["tag"], n=n, d=d, c=cn, nsites=info["nsites"], secs=info["secs"], teach=bits(env.acc(teach, lab)), neigh=bits(env.acc(neigh, goldf, NE_OK)), cal=bits(cal_acc()))
                if cn == "real": rec["annself"] = bits(env.acc(ann, lab, cs["_ix"])); rec["bytes"] = env.delta_stats()
                env.restore(); RES["runs"].append(rec)
        rr = [r for r in RES["runs"] if r["m"] == mt["tag"] and r["n"] == n]; f = lambda c, k: round(float(np.mean([np.mean(list(map(int, r[k]))) for r in rr if r["c"] == c])), 3) if any(r["c"] == c for r in rr) else -1
        log(mt["tag"], "n", n, "teach", f("real", "teach"), "shuf", f("shuffle", "teach"), "wrong", f("wrong", "teach"), "neigh", f("real", "neigh"), "cal", f("real", "cal"), round(time.time() - tm)); dump()
# continual: second feedback = another real draw (same label), first-round annotated items must stay correct
for tg in cfg.get("cont", []):
    mt = next(x for x in cfg["methods"] if x["tag"] == tg)
    for d in range(cfg.get("cont_draws", 4)):
        env.restore(); c1 = conds(12, 700 + d); env.apply(mt, c1["real"]); a1 = env.acc(ann, lab, c1["_ix"]); t1 = env.acc(teach, lab).mean(); n1 = env.acc(neigh, goldf, NE_OK).mean()
        rest = [i for i in range(len(ann)) if i not in c1["_ix"]]; ix2 = random.Random(900 + d).sample(rest, 12); env.apply(mt, [(ann[i], NEW) for i in ix2])
        RES["cont"].append(dict(m=tg, d=d, r1=dict(teach=float(t1), ann1=float(a1.mean()), neigh=float(n1)), r2=dict(teach=float(env.acc(teach, lab).mean()), ann1=float(env.acc(ann, lab, c1["_ix"]).mean()), ann2=float(env.acc(ann, lab, ix2).mean()), neigh=float(env.acc(neigh, goldf, NE_OK).mean()), cal=float(cal_acc().mean()))))
        env.restore()
    log("CONT", tg, RES["cont"][-1]); dump()
# real greedy generation validation on the first valmethod
for tg in cfg.get("valmethods", []):
    mt = next(x for x in cfg["methods"] if x["tag"] == tg); env.restore(); env.apply(mt, conds(12, 555)["real"]); V = []
    for nm, rows, g in (("teach", teach, lambda r: NEW), ("neigh", neigh, goldf)):
        idx = list(range(len(rows))) if nm == "teach" else NE_OK; idx = idx[::max(1, len(idx) // 120)][:120]; a1 = env.acc(rows, g if nm == "neigh" else lab, idx); go = env.gen_text([rows[i]["pid"] for i in idx], 8)
        a3 = np.array([int(x == g(rows[i])) for i, x in zip(idx, go)]); V.append(dict(m=tg, set=nm, agree=float((a1 == a3).mean()), acc_tf=float(a1.mean()), acc_gen=float(a3.mean())))
    RES.setdefault("val", []).extend(V); env.restore(); log("VAL", V); dump()
RES["done"] = True; dump(); log("DONE")
