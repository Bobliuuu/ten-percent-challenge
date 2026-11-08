"""analyze.py RES.json [RES2.json ...] -> markdown table (n=12 by default) sorted by (teach - shuffled) then neighbours then bytes. also writes summary.json"""
import json, sys, numpy as np, collections
rng = np.random.default_rng(0)
def bs(v, B=500):
    v = np.array(v, float); 
    if len(v) < 2: return (float(v.mean()) if len(v) else float("nan"),) * 2
    m = [rng.choice(v, len(v)).mean() for _ in range(B)]; return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))
def load(paths):
    out = []
    for p in paths:
        R = json.load(open(p)); R["_p"] = p; out.append(R)
    return out
def table(Rs, n=12, tagfilter=None):
    rows = []
    for R in Rs:
        meta = R["meta"]; ts = np.array([s == "clinc150" for s in meta["teach_src"]]); cells = [(s, g) for s, g in meta["neigh_meta"]]
        by = collections.defaultdict(list)
        for r in R["runs"]:
            if r["n"] == n: by[(r["m"], r["c"])].append(r)
        tags = sorted({k[0] for k in by}, key=lambda t: [m["tag"] for m in R["cfg"]["methods"]].index(t))
        for t in tags:
            g = lambda c: by.get((t, c), [])
            vec = lambda rs, k: [np.mean(list(map(int, r[k]))) for r in rs]
            real = g("real")
            if not real: continue
            tr = vec(real, "teach"); sh = vec(g("shuffle"), "teach"); wr = vec(g("wrong"), "teach"); ne = vec(real, "neigh"); cal = vec(real, "cal")
            # min cell over draws (mean per cell)
            cellacc = collections.defaultdict(list)
            for r in real:
                for (c, b) in zip(cells, r["neigh"]): cellacc[c].append(int(b))
            mincell = min(np.mean(v) for v in cellacc.values()) if cellacc else float("nan")
            worst = min(ne)
            tm = lambda rs, sel: [np.mean([int(b) for b, s in zip(r["teach"], ts) if s == sel]) for r in rs]
            xs_m = tm(g("real_massive"), True); xs_c = tm(g("real_clinc150"), False)     # trained on one source, tested on the OTHER source
            para = vec(real, "para") if "para" in real[0] else []; t2 = vec(real, "teach2") if "teach2" in real[0] else []
            ge = [r["bytes"] for r in real if "bytes" in r]; secs = np.mean([r["secs"] for r in real])
            rows.append(dict(model=R["cfg"]["model"].split("/")[-1] + ("-q4" if R["cfg"].get("q4") else ""), tag=t, nd=len(real), teach=np.mean(tr), ci=bs(tr), shuf=np.mean(sh) if sh else None, wrong=np.mean(wr) if wr else None,
                gap=np.mean(tr) - (np.mean(sh) if sh else np.nan), neigh=np.mean(ne), worst=worst, mincell=mincell, cal=np.mean(cal), para=np.mean(para) if para else None, teach2=np.mean(t2) if t2 else None,
                x_m2c=np.mean(xs_m) if xs_m else None, x_c2m=np.mean(xs_c) if xs_c else None, dense_mb=np.mean([b["bytes_dense"] for b in ge]) / 1e6 if ge else None, sparse_mb=np.mean([b["bytes_sparse99"] for b in ge]) / 1e6 if ge else None, secs=secs, nsites=np.mean([r["nsites"] for r in real])))
    return rows
f3 = lambda x: "-" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.3f}"
def md(rows):
    rows = sorted(rows, key=lambda r: (-(r["gap"] if r["gap"] == r["gap"] else -9), -r["neigh"]))
    h = "| model | method | draws | teach [CI] | shuffled | wrong | teach-shuf | neigh kept | worst draw | min cell | cal agree | paraphrase | teach2 | xsrc massive->clinc | xsrc clinc->massive | MB dense | MB sparse99 | s/edit |\n|" + "---|" * 18 + "\n"
    for r in rows:
        h += f"| {r['model']} | {r['tag']} | {r['nd']} | {f3(r['teach'])} [{r['ci'][0]:.2f},{r['ci'][1]:.2f}] | {f3(r['shuf'])} | {f3(r['wrong'])} | {f3(r['gap'])} | {f3(r['neigh'])} | {f3(r['worst'])} | {f3(r['mincell'])} | {f3(r['cal'])} | {f3(r['para'])} | {f3(r['teach2'])} | {f3(r['x_m2c'])} | {f3(r['x_c2m'])} | {f3(r['dense_mb'])} | {f3(r['sparse_mb'])} | {r['secs']:.0f} |\n"
    return h
if __name__ == "__main__":
    Rs = load(sys.argv[1:]); rows = table(Rs); print(md(rows))
    for R in Rs:
        for v in R.get("val", []): print("VAL", v["m"], v["set"], "agree", v["agree"], "tf", v["acc_tf"], "gen", v["acc_gen"])
        cc = collections.defaultdict(list)
        for c in R.get("cont", []): cc[(c["m"], c["mode"])].append(c)
        for (m, md_), L in cc.items():
            f = lambda k, rr: np.mean([np.mean(list(map(int, x[rr][k]))) for x in L]); print("CONT", m, md_, "r1 teach %.3f ann1 %.3f | r2 jukebox %.3f ann1 %.3f ann2 %.3f briefing %.3f neigh %.3f" % (f("teach", "r1"), f("ann1", "r1"), f("teach", "r2"), f("ann1", "r2"), f("ann2", "r2"), f("teachN", "r2"), f("neigh", "r2")))
    json.dump(rows, open("summary.json", "w"), default=float, indent=1)
