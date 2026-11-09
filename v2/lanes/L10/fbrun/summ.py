"""python summ.py RES.json -> markdown table: teach / neighbours per arm pooled over draws (Wilson 95% lower), per-draw min/max, per-source teach, generation agreement."""
import json, sys, math, collections, numpy as np
def wl(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); r = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)); return (c - r) / d
d = json.load(open(sys.argv[1])); print("base", d["base"]); src = d["meta"]["teach_src"]
by = collections.defaultdict(list)
for r in d["runs"]: by[(r["m"], r["c"])].append(r)
for (m, c), rs in by.items():
    row = {}
    for k in ("teach", "neigh", "cal"):
        a = np.array([[int(x) for x in r[k]] for r in rs]); row[k] = (a.mean(), wl(int(a.sum()), a.size), a.mean(1).min(), a.mean(1).max())
    s = {}
    a = np.array([[int(x) for x in r["teach"]] for r in rs])
    for sn in sorted(set(src)): idx = [i for i, x in enumerate(src) if x == sn]; s[sn] = round(float(a[:, idx].mean()), 3)
    print(f"| {m} | {c} | {len(rs)} | teach {row['teach'][0]:.3f} [{row['teach'][1]:.3f}] draws {row['teach'][2]:.3f}-{row['teach'][3]:.3f} | neigh {row['neigh'][0]:.4f} [{row['neigh'][1]:.4f}] min {row['neigh'][2]:.3f} | cal {row['cal'][0]:.3f} | by src {s} |")
print("val", d.get("val")); print("cont", [ (x["m"], x["r1"], x["r2"]) for x in d.get("cont", [])][:2])
