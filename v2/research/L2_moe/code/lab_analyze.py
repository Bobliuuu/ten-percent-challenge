"""usage: lab_analyze.py lab235_*.json  -> per task retention vs base (teacher-forced argmax==label), overall + role + kind cells, Wilson lo, strict gates"""
import json, sys, math, collections
def wil(k, n, z=1.96):
    if n == 0: return 0.0
    p = k/n; d = 1+z*z/n; c = p+z*z/(2*n); h = z*math.sqrt(p*(1-p)/n+z*z/(4*n*n)); return (c-h)/d
R = {}
for f in sys.argv[1:]: R.update(json.load(open(f)))
B = R["base"]; tasks = sorted({v["task"] for v in B.values()})
print("base acc per task:", {t: round(sum(v["ok"] for v in B.values() if v["task"] == t)/sum(1 for v in B.values() if v["task"] == t), 3) for t in tasks})
for c, C in R.items():
    if c == "base": continue
    print("\n==", c)
    for t in tasks:
        g = collections.defaultdict(lambda: [0, 0])
        for i, b in B.items():
            if b["task"] == t and b["ok"]:
                for key in ("ALL", "role:"+b["source"], "kind:"+b["kind"], "cell:"+b["source"]+"/"+b["kind"]): g[key][1] += 1; g[key][0] += C[i]["ok"]
        cells = {k: v for k, v in g.items() if k.startswith("cell:") and v[1] >= 10}
        a = g["ALL"]; ret = a[0]/a[1]; wk, wv = min(cells.items(), key=lambda kv: kv[1][0]/kv[1][1]) if cells else ("-", [0, 1])
        ok = ret >= .98 and wil(*a) >= .98 and wv[0]/wv[1] >= .95
        print(f"{t:8s} ret {ret:.3f} (lo {wil(*a):.3f}; {a[0]}/{a[1]}) worst {wk} {wv[0]/wv[1]:.3f}({wv[1]}) PASS={ok} | "+"; ".join(f"{k} {v[0]/v[1]:.2f}({v[1]})" for k, v in sorted(g.items()) if k.startswith("role")))
