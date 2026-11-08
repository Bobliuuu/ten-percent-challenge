"""tables from run_fbx.py output.  python analyze_fbx.py OUT.json [sweep|final|policy]"""
import sys, json, numpy as np, collections
R = json.load(open(sys.argv[1])); runs = R["runs"]
m = lambda L: round(float(np.mean(L)), 4) if len(L) else None
def agg(rs, k): 
    v = [np.mean(r["res"][k]["ok"]) for r in rs if k in r["res"] and r["res"][k]["ok"]]; return (m(v), m([min(v)]) if v else None, len(v))
def ci(v, B=2000, seed=0):
    if len(v) < 2: return None
    rr = np.random.default_rng(seed); bs = [np.mean(rr.choice(v, len(v))) for _ in range(B)]; return [round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3)]
print("sizes", R["sizes"], "base teach", R["base"])
if R["mode"] == "sweep":
    by = collections.defaultdict(list)
    for r in runs: by[r["tag"]].append(r)
    for tag, rs in by.items():
        re_ = [r for r in rs if r["arm"] == "real"]; sh = [r for r in rs if r["arm"] == "shuffled"]
        print(f"{tag:44s} teach {agg(re_,'teach_sel')[0]} shuf {agg(sh,'teach_sel')[0]} neigh_sel {agg(re_,'neigh_sel')[0]} (worst {agg(re_,'neigh_sel')[1]}) hp {agg(re_,'neigh_sel_hp')[0]} MB {re_[0]['bytes']/1e6:.1f} edit {re_[0]['secs']}s")
else:
    by = collections.defaultdict(list)
    for r in runs: by[(r["arm"], r["n"])].append(r)
    for (arm, n), rs in by.items():
        print(f"{arm:9s} n={n:2d} draws={len(rs):2d} " + " ".join(f"{k}={agg(rs,k)[0]}(min {agg(rs,k)[1]})" for k in ("teach_rep", "teach_bbc", "neigh_rep", "old_teach") if any(k in r["res"] for r in rs)) + f" MB={rs[0]['bytes']/1e6:.1f}")
