"""score_s7.py BASE.json CAND.json : retention of base-correct SKILL7 items (real generated word), per cell (src,kind), Wilson LB."""
import json, sys, math, collections
it = {json.loads(l)["id"]: json.loads(l) for l in open(__file__.replace("score_s7.py", "data/skill7_probe.jsonl"))}
w = lambda o: (o.split() or [""])[0].strip(".,:;!\"'").lower()
B, C = json.load(open(sys.argv[1])), json.load(open(sys.argv[2]))
def wil(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)); return (c - h) / d
ok = lambda R, i: w(R[i]) == it[i]["answer"].lower()
print("base acc %.4f cand acc %.4f n=%d" % (sum(ok(B, i) for i in it) / len(it), sum(ok(C, i) for i in it) / len(it), len(it)))
cells = collections.defaultdict(lambda: [0, 0]); T = [0, 0]
for i in it:
    if ok(B, i):
        for k in (it[i]["src"] + "|" + it[i]["kind"], "kind:" + it[i]["kind"]): cells[k][1] += 1; cells[k][0] += ok(C, i)
        T[1] += 1; T[0] += ok(C, i)
print("retention %.4f LB %.4f n=%d" % (T[0] / T[1], wil(*T), T[1]))
for k, v in sorted(cells.items()): print("  %-28s %.3f (%d)" % (k, v[0] / v[1], v[1]))
