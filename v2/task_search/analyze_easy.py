"""python analyze_easy.py <raw dir>: per model, per task: base acc, retention (real gen) with Wilson lo, worst (src x kind) cell >=8 base-correct, per-kind pooled min"""
import json, glob, math, collections, sys, os
def wl(k, n, z=1.96):
    if n == 0: return 0
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); r = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)); return (c - r) / d
out = []
for bf in sorted(glob.glob(sys.argv[1] + "/ct_*_e_bf16.json")):
    tag = os.path.basename(bf)[3:-len("_e_bf16.json")]; B = {r["id"]: r for r in json.load(open(bf))["rows"]}
    for qf in sorted(glob.glob(sys.argv[1] + f"/ct_{tag}_e_r*.json")):
        d = json.load(open(qf)); Q = {r["id"]: r for r in d["rows"]}; name = os.path.basename(qf)
        out.append(f"\n### {tag}  {name}  ratio={d['ratio']*100:.3f}%  body_bpw={d['body_bpw']:.2f}")
        out.append("| task | n | base acc | base-correct | retention (Wilson lo) | worst src x kind (n) | worst pooled kind (n) |\n|---|---|---|---|---|---|---|")
        tasks = sorted(set(r["task"] for r in B.values()))
        for t in tasks + ["ALL"]:
            ids = [i for i, r in B.items() if (t == "ALL" or r["task"] == t)]; bc = [i for i in ids if B[i]["gen_correct"]]
            k = sum(Q[i]["gen_correct"] for i in bc); cell = collections.defaultdict(lambda: [0, 0]); kind = collections.defaultdict(lambda: [0, 0])
            for i in bc:
                r = B[i]; c = cell[(r["src"], r["kind"])]; c[1] += 1; c[0] += Q[i]["gen_correct"]; c = kind[(r["task"], r["kind"])]; c[1] += 1; c[0] += Q[i]["gen_correct"]
            def worst(D): 
                v = [(a / b, key, b) for key, (a, b) in D.items() if b >= 8]; return min(v) if v else (None, None, 0)
            w, wk, wn = worst(cell); pk, pkk, pn = worst(kind)
            out.append(f"| {t} | {len(ids)} | {len(bc)/max(1,len(ids)):.3f} | {len(bc)} | {k/max(1,len(bc)):.3f} [{wl(k,len(bc)):.3f}] | {w if w is None else f'{w:.2f}'} {wk} ({wn}) | {pk if pk is None else f'{pk:.2f}'} {pkk} ({pn}) |")
print("\n".join(out))
