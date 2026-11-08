"""agg_fb.py fb.json : mean/min per (arm, n)"""
import json, sys, collections, statistics
r = json.load(open(sys.argv[1])); g = collections.defaultdict(list)
for x in r["runs"]: g[(x["arm"], x.get("n"))].append(x)
for k, v in sorted(g.items(), key=str):
    print(k, len(v), "teach mean %.3f min %.3f | neigh mean %.3f min %.3f | min_cell mean %.3f min %.3f" % (*[f(y[m] for y in v) for m in ("teach", "neigh", "min_cell") for f in (statistics.mean, min)],))
