"""usage: python analyze_recipe.py results/fb_s7_recipe.json -> teach (all/same-source/cross-source), neighbours (exam-neighbour all/clean_only, old-skill dev), per method/n/cond, mean over draws"""
import sys, json, os, collections, numpy as np
sys.path.insert(0, os.path.abspath("../../exam")); from loader import load
d = json.load(open(sys.argv[1]))
T = {r["id"]: r for r in load("SKILL7", "teach")}; X = {r["id"]: r for r in load("SKILL7", "exam")}
tg = d["teach_grp"]; nm = d["neigh_meta"]
cross = np.array([T[g]["src"] in ("hwu64", "mtop") for g in tg]); 
isx = np.array([m[2] in X for m in nm]); clean = np.array([(m[2] in X and X[m[2]].get("tier", "clean") == "clean") for m in nm])
agg = collections.defaultdict(list)
for r in d["runs"]:
    t = np.array([int(c) for c in r["teach"]]); n = np.array([int(c) for c in r["neigh"]])
    agg[(r["m"], r["n"], r["c"])].append(dict(teach=t.mean(), same=t[~cross].mean(), cross=t[cross].mean(), neigh=n.mean(), nexam=n[isx].mean(), nclean=n[clean].mean(), ndev=n[~isx].mean()))
print("n_teach", len(tg), "cross", int(cross.sum()), "| neigh base-correct", len(nm), "exam", int(isx.sum()), "clean exam", int(clean.sum()), "dev", int((~isx).sum()))
print("method n cond draws | teach same cross | neigh exam(all) exam(clean) dev")
for k, v in sorted(agg.items()):
    f = lambda key: np.mean([x[key] for x in v])
    print(f"{k[0]:9s} {k[1]:3d} {k[2]:8s} {len(v)} | {f('teach'):.3f} {f('same'):.3f} {f('cross'):.3f} | {f('nexam'):.4f} {f('nclean'):.4f} {f('ndev'):.4f}")
print("validity", json.dumps(d.get("validity"))[:1500])
