"""retention of candidate vs base on the dev set: python analyze_ret.py base.json cand.json [dev_inj.jsonl]. Retention = P(cand correct | base correct).
Reports overall, per kind (source x class), per class, per source, Wilson 95% lower bound, group-cluster bootstrap lower bound (2000 draws), format failures, base/cand accuracy."""
import json, sys, math, random, collections
def wil(k, n, z=1.96):
    if n == 0: return 0.0
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)); return (c - h) / d
B = {r["id"]: r for r in json.load(open(sys.argv[1]))["rows"]}; C = {r["id"]: r for r in json.load(open(sys.argv[2]))["rows"]}
dev = {}
for l in open(sys.argv[3] if len(sys.argv) > 3 else "dev_inj.jsonl"):
    d = json.loads(l); dev[d["id"]] = d
ids = [i for i in B if B[i]["correct"]]; n = len(ids)
def cells(i): r = B[i]; return {"ALL": 1, r["kind"]: 1, "class:" + r["gold"]: 1, "src:" + r["source"]: 1}
agg = collections.defaultdict(lambda: [0, 0])
for i in ids:
    for k in cells(i): agg[k][1] += 1; agg[k][0] += C[i]["correct"]
print("n_items", len(B), "base acc %.4f cand acc %.4f" % (sum(r["correct"] for r in B.values()) / len(B), sum(C[i]["correct"] for i in B) / len(B)), "cand format-fail", sum(C[i]["pred"] is None for i in B))
for k, (a, b) in sorted(agg.items()): print("%-22s %5d/%-5d ret %.4f  wilson_lo %.4f" % (k, a, b, a / b, wil(a, b)))
grp = collections.defaultdict(list)
for i in ids: grp[dev[i]["group"]].append(i)
gl = list(grp.values()); random.seed(0); bs = []
for _ in range(2000):
    s = [i for g in random.choices(gl, k=len(gl)) for i in g]; bs.append(sum(C[i]["correct"] for i in s) / len(s))
bs.sort(); print("cluster-bootstrap overall ret 2.5%% = %.4f  (groups %d)" % (bs[50], len(gl)))
