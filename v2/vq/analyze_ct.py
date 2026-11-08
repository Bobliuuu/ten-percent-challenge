"""retention from dense_core ct_*.json (teacher-forced argmax == greedy reproduction): python analyze_ct.py BASE.json CAND.json"""
import json, math, collections, sys
def wil(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n; return (p + z * z / (2 * n) - z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / d
b = json.load(open(sys.argv[1]))["rows"]; c = {r["id"]: r for r in json.load(open(sys.argv[2]))["rows"]}
cells = collections.defaultdict(lambda: [0, 0]); acc = [0, 0, 0]
for r in b:
    acc[0] += r["correct"]; acc[1] += c[r["id"]]["correct"]; acc[2] += 1
    if not r["correct"]: continue
    for key in ("ALL", "kind:" + r["kind"], "src:" + r["src"], r["src"] + "_" + r["kind"]):
        cells[key][1] += 1; cells[key][0] += c[r["id"]]["correct"]
print("base acc", acc[0] / acc[2], "cand", acc[1] / acc[2])
for k in sorted(cells, key=lambda k: (k != "ALL", k)): a, n = cells[k]; print(f"  {k:28s} {a}/{n} ret {a/n:.4f} lo {wil(a,n):.4f}")
