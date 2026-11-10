"""Select the carrier config from DEV logs only. Usage: pick.py dev_runs.json. Rule (declared before looking): admissible = max shuffle teach over draws <= 0.02, min real-draw neighbours >= empty-0.01, min shuffle neighbours >= empty-0.01; pick max mean real teach; if none admissible pick max (mean real teach - max shuffle teach)."""
import json, sys, collections
runs = json.load(open(sys.argv[1])); emp = runs["empty"]["neigh_ret"]; G = collections.defaultdict(lambda: collections.defaultdict(list)); cfg = {}
for r in runs["runs"]: k = r["gi_key"]; G[k][r["cond"]].append(r); cfg[k] = r["grid"]
rows = []
for k, d in G.items():
    rt = [x["teach_acc"] for x in d["real"]]; st = [x["teach_acc"] for x in d["shuffle"]]; rn = [x["neigh_ret"] for x in d["real"]]; sn = [x["neigh_ret"] for x in d["shuffle"]]
    if len(rt) < 4 or len(st) < 4: continue
    mt = sum(rt) / len(rt); adm = max(st) <= 0.02 and min(rn) >= emp - 0.01 and min(sn) >= emp - 0.01
    rows.append(dict(cfg=cfg[k], real_mean=mt, real=rt, shuf_max=max(st), shuf_mean=sum(st) / len(st), neigh_min=min(rn), shuf_neigh_min=min(sn), admissible=adm, gap=mt - max(st)))
rows.sort(key=lambda r: (-r["admissible"], -r["real_mean"] if r["admissible"] else -r["gap"]))
for r in rows: print("ADM" if r["admissible"] else "   ", r["cfg"], "real %.3f shufmax %.3f shufmean %.3f neighmin %.3f shufneighmin %.3f" % (r["real_mean"], r["shuf_max"], r["shuf_mean"], r["neigh_min"], r["shuf_neigh_min"]))
json.dump(dict(empty_neigh=emp, ranked=rows, chosen=rows[0]["cfg"]), open("dev_pick.json", "w"), indent=1); print("CHOSEN", rows[0]["cfg"])
