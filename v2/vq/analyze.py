"""retention of base-correct items per (source x kind) cell, Wilson 95% lower bound. usage: python analyze.py BASE.json CAND.json"""
import sys, json, math, collections
def wil(k, n, z=1.96):
    if n == 0: return float("nan")
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)); return (c - h) / d
b = json.load(open(sys.argv[1])); c = json.load(open(sys.argv[2]))
for d in (b, c):
    if "inj" in d: d["real_inj"], d["real_news"] = d["inj"], d["news"]
for task in ("real_inj", "real_news"):
    br = {r["id"]: r for r in b[task]}; cr = {r["id"]: r for r in c[task]}
    cells = collections.defaultdict(lambda: [0, 0]); 
    for i, r in br.items():
        if not r["correct"]: continue
        for key in ("ALL", r["source"] + "_" + r["kind"], "kind:" + r["kind"], "src:" + r["source"]):
            cells[key][1] += 1; cells[key][0] += int(cr[i]["correct"])
    print(f"== {task}: base acc {sum(r['correct'] for r in br.values())/len(br):.4f} cand acc {sum(r['correct'] for r in cr.values())/len(cr):.4f}")
    for k in sorted(cells, key=lambda k: (k != "ALL", k)):
        a, n = cells[k]; print(f"  {k:28s} {a}/{n} ret {a/n:.4f} wilson_lo {wil(a,n):.4f}")
