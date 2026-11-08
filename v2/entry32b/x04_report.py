"""X04a summary: python x04_report.py  (reads results/real_e32_base.json + results/x04/real_x04_<ARM>_s<seed>.json). Writes results/x04/X04A_TABLE.md"""
import json, glob, hashlib, collections, math, sys, os
H = os.path.dirname(os.path.abspath(__file__)); R = H + "/results/"
dev = {}; [dev.__setitem__(d["id"], d) for d in map(json.loads, open(H + "/dev_inj.jsonl"))]
B = {r["id"]: r for r in json.load(open(R + "real_e32_base.json"))["rows"]}
# INJR-dev subset = dev_S1 + dev_S3 + (200 inj + 200 ben of audit_S2 chosen by sha256('injr'+id)) intersected with this dev set
A = [json.loads(l) for l in open(H + "/../tournament/INJ/data/audit_S2.jsonl")]; key = lambda r: hashlib.sha256(("injr" + r["id"]).encode()).hexdigest()
s2 = {r["id"] for r in sorted([r for r in A if r["gold"] == "injection"], key=key)[:200] + sorted([r for r in A if r["gold"] == "benign"], key=key)[:200]}
D1 = {json.loads(l)["id"] for l in open(H + "/../tournament/INJ/data/dev_S1.jsonl")}; D3 = {json.loads(l)["id"] for l in open(H + "/../tournament/INJ/data/dev_S3.jsonl")}
injr = {i for i in B if i in D1 or i in D3 or i in s2}
def wil(k, n, z=1.96):
    if n == 0: return 0.0
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)); return (c - h) / d
def stats(C, ids):
    ids = [i for i in ids if B[i]["correct"]]; cells = collections.defaultdict(lambda: [0, 0]); tot = [0, 0]
    for i in ids:
        for k in (B[i]["kind"], "class:" + B[i]["gold"]): cells[k][1] += 1; cells[k][0] += C[i]["correct"]
        tot[1] += 1; tot[0] += C[i]["correct"]
    return tot, cells
arms = collections.defaultdict(dict)
for f in sorted(glob.glob(R + "x04/real_x04_*_s*.json")):
    n = os.path.basename(f)[9:-5]; a, s = n.rsplit("_s", 1); arms[a][s] = {r["id"]: r for r in json.load(open(f))["rows"]}
out = ["| arm | seeds | scope | n base-correct | flip % per seed | mean retention | pooled Wilson lo | worst gated cell (n>=150) | min class |", "|---|---|---|---|---|---|---|---|---|"]
for scope, ids in (("dev_inj 4313", list(B)), ("INJR-dev subset", sorted(injr))):
    for a, ss in sorted(arms.items()):
        flips = []; k = n_ = 0; worst = (2, ""); mc = (2, ""); nb = 0
        cellsum = collections.defaultdict(lambda: [0, 0])
        for s, C in sorted(ss.items()):
            tot, cells = stats(C, ids); flips.append(100 * (1 - tot[0] / tot[1])); k += tot[0]; n_ += tot[1]; nb = tot[1]
            for c, v in cells.items(): cellsum[c][0] += v[0]; cellsum[c][1] += v[1]
        for c, v in cellsum.items():
            r = v[0] / v[1]; nn = v[1] / len(ss)
            if c.startswith("class:"): mc = min(mc, (r, c))
            elif nn >= 150: worst = min(worst, (r, c))
        out.append(f"| {a} | {','.join(sorted(ss))} | {scope} | {nb} | {', '.join('%.2f' % f for f in flips)} | {k / n_:.4f} | {wil(k, n_):.4f} | {worst[0]:.3f} {worst[1]} | {mc[0]:.3f} {mc[1]} |")
open(R + "x04/X04A_TABLE.md", "w").write("\n".join(out) + "\n"); print("\n".join(out))
