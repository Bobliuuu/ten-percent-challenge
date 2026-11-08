import json,sys,numpy as np,collections
for p in sys.argv[1:]:
    R=json.load(open(p)); g=collections.defaultdict(list)
    for r in R["runs"]: g[(r["m"],r["n"],r["c"])].append(r)
    for (m,n,c),rs in g.items():
        f=lambda k:np.mean([np.mean(list(map(int,r[k]))) for r in rs])
        print("%-26s n=%2d %-11s site=%.2f teach=%.3f ood=%.3f neigh=%.4f nbnew/fail=%.1f/%.1f nd=%d"%(m,n,c,np.mean([r["site_ok"] for r in rs]),f("teach"),f("ood"),f("neigh"),np.mean([r.get("nb_new",0) for r in rs]),np.mean([r.get("nb_fail",0) for r in rs]),len(rs)))
