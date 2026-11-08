"""python analyze.py RESULTS_DIR TAG BASE_NAME Q_NAME...  retention of base-correct items under REAL greedy generation, Wilson lower bound, per (task,kind) and (src,kind) cell minima."""
import json, math, sys, collections
def wl(k, n, z=1.96):
    if n == 0: return 0.0
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); r = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)); return (c - r) / d
D, tag, bn = sys.argv[1:4]
B = {r["id"]: r for r in json.load(open(f"{D}/ct_{tag}_{bn}.json"))["rows"]}
out = {}
for qn in sys.argv[4:]:
    d = json.load(open(f"{D}/ct_{tag}_{qn}.json")); Q = {r["id"]: r for r in d["rows"]}; res = dict(ratio=d["ratio"], body_bpw=d["body_bpw"], bytes=d["bytes"], base_bytes=d["base_bytes"])
    for t in sorted(set(r["task"] for r in B.values())) + ["ALL"]:
        ids = [i for i, r in B.items() if t == "ALL" or r["task"] == t]; bc = [i for i in ids if B[i]["gen_correct"]]; k = sum(Q[i]["gen_correct"] for i in bc)
        kind = collections.defaultdict(lambda: [0, 0]); cell = collections.defaultdict(lambda: [0, 0])
        for i in bc: r = B[i]; kind[r["kind"]][1] += 1; kind[r["kind"]][0] += Q[i]["gen_correct"]; cell[(r["src"], r["kind"])][1] += 1; cell[(r["src"], r["kind"])][0] += Q[i]["gen_correct"]
        mn = lambda D_: min(((a / b, key, b, a) for key, (a, b) in D_.items() if b >= 8), default=None)
        res[t] = dict(n=len(ids), base_acc=len(bc) / max(1, len(ids)), base_correct=len(bc), kept=k, retention=k / max(1, len(bc)), wilson_lo=wl(k, len(bc)), worst_kind=mn(kind), worst_src_kind=mn(cell),
                      kinds={f"{a}": [x, y, round(x / y, 3)] for a, (x, y) in sorted(kind.items())})
    out[qn] = res
    print(qn, f"ratio={res['ratio']*100:.3f}% bpw={res['body_bpw']:.2f}")
    for t in sorted(k for k in res if k not in ("ratio", "body_bpw", "bytes", "base_bytes")):
        v = res[t]; print(f"  {t:8s} base {v['base_acc']:.3f} ({v['base_correct']}/{v['n']}) retention {v['retention']:.4f} [{v['wilson_lo']:.4f}] worst kind {v['worst_kind']} worst src x kind {v['worst_src_kind']}")
json.dump(out, open(f"{D}/analysis_{tag}.json", "w"), indent=1)
