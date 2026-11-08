"""summarise result JSONs: python analyze_l1p.py results/swA.json [...]  -> per (tag, arm): mean over draws of every eval set, bytes."""
import sys, json, collections, numpy as np
for f in sys.argv[1:]:
    R = json.load(open(f)); g = collections.defaultdict(list)
    for r in R["runs"]: g[(r["tag"], r["arm"], r["n"])].append(r)
    print("==", f)
    for (t, a, n), rs in g.items():
        keys = sorted({k for r in rs for k in r["res"]}); s = {k: round(float(np.mean([np.mean(r["res"][k]["ok"]) for r in rs if k in r["res"]])), 3) for k in keys}
        mn = {k: round(float(min(np.mean(r["res"][k]["ok"]) for r in rs if k in r["res"])), 3) for k in keys if k.startswith("neigh")}
        print(f"{t:24s} {a:22s} n={n:<3d} d={len(rs)} bytes={rs[0]['bytes']:>9d}", s, ("min " + str(mn)) if mn else "")
