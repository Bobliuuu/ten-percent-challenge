"""L11 ablation runner. usage: python run_abl.py CFG.json OUTDIR
cfg: model, q4, ndraw, ns, conds, methods[{tag,carrier,solver,layers,m,lam,awt,...}], valmethods[tags], cont[{tag,mode,n,draws}], xsrc_n, tag_prefix"""
import json, sys, os, time, random, collections, numpy as np, torch
from fblib import Env, OLD, NEWS
cfg = json.load(open(sys.argv[1])); OUT = sys.argv[2]; os.makedirs(OUT, exist_ok=True)
T0 = time.time(); log = lambda *a: print(f"[{time.time()-T0:6.0f}s]", *a, flush=True)
env = Env(cfg["model"], cfg.get("data", "data"), q4=cfg.get("q4", False), ncal=cfg.get("ncal", 300), log=log)
R = env.R; NEW1 = NEWS["Music"]; NEW2 = NEWS["News"]
bits = lambda a: "".join(map(str, a.tolist()))
# ---- paraphrases of annotated items (base model, greedy)
PP = "Rewrite the following request in different words, keeping the same meaning. Output only the rewritten request.\n\nRequest: {X}"
pp = env.gen_text([env.tok(env.chat(PP.replace("{X}", r["text"])), add_special_tokens=False).input_ids for r in R["ann_Music"]], 60)
PARA = [dict(text=t or r["text"], gold="Music") for t, r in zip(pp, R["ann_Music"])]
for r in PARA: r["pid"] = env.pid(r["text"])
log("para sample", PARA[0]["text"][:100])
# ---- base
BASE_LAB = env.gen_text([r["pid"] for r in R["ann_Music"]], 6); log("base labels on Music ann pool", collections.Counter(BASE_LAB).most_common(6))
BASE_LAB_N = env.gen_text([r["pid"] for r in R["ann_News"]], 6); log("base labels on News ann pool", collections.Counter(BASE_LAB_N).most_common(6))
neigh = R["neigh"]; teach = R["teach_Music"]; teach2 = R["calnew_Music"][::2][:150]; teachN = R["teach_News"]; cal_eval = env.R["calib_old"][300:450]
lab_of = lambda lab: (lambda r: lab)
base_neigh = env.acc(neigh, lambda r: r["gold"]); NE_OK = np.where(base_neigh == 1)[0].tolist()
base_cal = env.gen_text([r["pid"] for r in cal_eval], 6); CAL_OK = [i for i, r in enumerate(cal_eval) if base_cal[i] == r["gold"]]
base = dict(teach=float(env.acc(teach, lab_of(NEW1)).mean()), teach2=float(env.acc(teach2, lab_of(NEW1)).mean()), teachN=float(env.acc(teachN, lab_of(NEW2)).mean()), neigh=float(base_neigh.mean()))
log("BASE", base, "base-correct neighbours", len(NE_OK), "cal-eval ok", len(CAL_OK))
cal_gold = {i: cal_eval[i]["gold"] for i in CAL_OK}
cal_acc = lambda: env.acc(cal_eval, lambda r: r["gold"], CAL_OK)
meta = dict(neigh_meta=[(neigh[i]["src"], neigh[i]["gold"]) for i in NE_OK], teach_src=[r["src"] for r in teach], teach2_src=[r["src"] for r in teach2], teachN_n=len(teachN))
# ---- draws
poolM = list(range(len(R["ann_Music"]))); poolNw = list(range(len(R["ann_News"])))
def draw(rows_pool, n, seed, src=None, exclude=()):
    ids = [i for i in rows_pool if (src is None or R["ann_Music"][i]["src"] == src) and i not in exclude]
    return random.Random(seed).sample(ids, min(n, len(ids)))
def conds_for(n, d):
    out = {}; ix = draw(poolM, n, 1000 * n + d); rr = random.Random(5000 + 1000 * n + d)
    out["real"] = (ix, [(R["ann_Music"][i], NEW1) for i in ix])
    out["wrong"] = (ix, [(R["ann_Music"][i], rr.choice([l for l in OLD if l != BASE_LAB[i]])) for i in ix])
    pool_o = R["calib_old"][450:]; sh = rr.sample(range(len(pool_o)), n); out["shuffle"] = (sh, [(pool_o[i], NEW1) for i in sh])
    for sname in ("massive", "clinc150"):
        jx = draw(poolM, n, 3000 * n + d, src=sname); out["real_" + sname] = (jx, [(R["ann_Music"][i], NEW1) for i in jx])
    return out
def evaluate(rec, ix_ann=None, full=True):
    rec["teach"] = bits(env.acc(teach, lab_of(NEW1))); rec["neigh"] = bits(env.acc(neigh, lambda r: r["gold"], NE_OK)); rec["cal"] = bits(cal_acc())
    if full: rec["teach2"] = bits(env.acc(teach2, lab_of(NEW1)))
    return rec
def acc_items(items):
    return float(np.mean([env.acc([r], lambda r_, l=l: l)[0] for r, l in items]))
def apply_any(mt, items):
    """autocal: choose the margin m from the feedback itself (annotated items + calibration prompts only): smallest m with annotated acc>=0.95 and calibration agreement>=0.985"""
    if not mt.get("autocal"): return env.apply(mt, items)
    t0 = time.time(); chosen = None; fb = None
    for m in mt["autocal"]:
        env.restore(); env.apply(dict(mt, m=m), items); a = acc_items(items); c = float(cal_acc().mean())
        if c >= 0.985: fb = m
        if c >= 0.985 and a >= 0.95: chosen = m; break
    chosen = chosen or fb or mt["autocal"][0]; env.restore(); info = env.apply(dict(mt, m=chosen), items); info["m_chosen"] = chosen; info["secs"] = time.time() - t0; return info
RES = dict(cfg=cfg, base=base, meta=meta, runs=[], cont=[], val=[], bytes={})
def dump(): json.dump(RES, open(f"{OUT}/res.json", "w")); 
prev = None
for mt in cfg["methods"]:
    tag = mt["tag"]; key = (mt["carrier"], tuple(mt.get("layers", [])))
    if prev != key: env.clear_cache(); prev = key
    NS = mt.get("ns", cfg["ns"]); ND = mt.get("ndraw", cfg["ndraw"]); CD = mt.get("conds", cfg["conds"]); tm = time.time()
    for n in NS:
        for d in range(ND):
            cs = conds_for(n, d)
            for cn in CD:
                if cn in ("real_massive", "real_clinc150") and n != cfg.get("xsrc_n", 12): continue
                ix, items = cs[cn]; env.restore(); info = apply_any(mt, items)
                rec = dict(m=tag, n=n, d=d, c=cn, nsites=info["nsites"], secs=info["secs"], mch=info.get("m_chosen")); evaluate(rec, full=(cn == "real"))
                if cn.startswith("real"):
                    rec["annself"] = bits(env.acc(R["ann_Music"], lab_of(NEW1), ix)); rec["para"] = bits(env.acc(PARA, lab_of(NEW1), ix))
                    if d < 3: rec["bytes"] = env.delta_stats()
                env.restore(); RES["runs"].append(rec)
        rr = [r for r in RES["runs"] if r["m"] == tag and r["n"] == n]; f = lambda c, k: round(float(np.mean([np.mean(list(map(int, r[k]))) for r in rr if r["c"] == c])), 3) if any(r["c"] == c for r in rr) else -1
        log(tag, "n", n, "teach real", f("real", "teach"), "shuf", f("shuffle", "teach"), "wrong", f("wrong", "teach"), "neigh", f("real", "neigh"), "cal", f("real", "cal"), "mssv", f("real_massive", "teach"), "clinc", f("real_clinc150", "teach"), "secs", round((time.time() - tm), 0))
        dump()
# ---- real-generation validation
for tg in cfg.get("valmethods", []):
    mt = next(x for x in cfg["methods"] if x["tag"] == tg)
    for dd in range(2):
        ix, items = conds_for(12, 555 + dd)["real"]; env.restore(); env.apply(mt, items)
        for nm, rows, lab in (("teach", teach, None), ("neigh", neigh, True)):
            idx = list(range(len(rows))) if lab is None else NE_OK; idx = idx[::max(1, len(idx) // 100)][:100]
            a1 = env.acc(rows, (lambda r: NEW1) if lab is None else (lambda r: r["gold"]), idx); go = env.gen_text([rows[i]["pid"] for i in idx], 8)
            a3 = np.array([int(g == (NEW1 if lab is None else rows[i]["gold"])) for i, g in zip(idx, go)])
            RES["val"].append(dict(m=tg, draw=dd, set=nm, agree=float((a1 == a3).mean()), acc_tf=float(a1.mean()), acc_gen=float(a3.mean()), sample=collections.Counter(go).most_common(3)))
        env.restore()
    log("VAL", tg, RES["val"][-4:]); dump()
# ---- continual (apply feedback twice)
for c in cfg.get("cont", []):
    mt = next(x for x in cfg["methods"] if x["tag"] == c["tag"]); n = c.get("n", 12)
    for d in range(c.get("draws", 5)):
        env.restore(); ix1 = draw(poolM, n, 8000 + d); it1 = [(R["ann_Music"][i], NEW1) for i in ix1]
        env.apply(mt, it1); r1 = evaluate(dict(), None, full=False); r1["ann1"] = bits(env.acc(R["ann_Music"], lab_of(NEW1), ix1))
        if c["mode"] == "same":
            ix2 = draw(poolM, n, 9000 + d, exclude=set(ix1)); it2 = [(R["ann_Music"][i], NEW1) for i in ix2]
        else:
            ix2 = random.Random(9000 + d).sample(poolNw, min(n, len(poolNw))); it2 = [(R["ann_News"][i], NEW2) for i in ix2]
        info = env.apply(mt, it2); r2 = evaluate(dict(), None, full=False); r2["ann1"] = bits(env.acc(R["ann_Music"], lab_of(NEW1), ix1))
        r2["ann2"] = bits(env.acc(R["ann_Music" if c["mode"] == "same" else "ann_News"], lab_of(NEW1 if c["mode"] == "same" else NEW2), ix2)); r2["teachN"] = bits(env.acc(teachN, lab_of(NEW2))); r2["nsites2"] = info["nsites"]
        RES["cont"].append(dict(m=c["tag"], mode=c["mode"], n=n, d=d, r1=r1, r2=r2)); env.restore()
    cc = [x for x in RES["cont"] if x["m"] == c["tag"] and x["mode"] == c["mode"]]; f = lambda k, rr: round(float(np.mean([np.mean(list(map(int, x[rr][k]))) for x in cc])), 3)
    log("CONT", c["tag"], c["mode"], "r1 teach", f("teach", "r1"), "r1 ann1", f("ann1", "r1"), "| r2 teach(Jukebox)", f("teach", "r2"), "ann1", f("ann1", "r2"), "ann2", f("ann2", "r2"), "teachN(Briefing)", f("teachN", "r2"), "neigh", f("neigh", "r2")); dump()
RES["done"] = True; dump(); open(f"{OUT}/DONE", "w").write("ok"); log("DONE")
