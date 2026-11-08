"""python analyze_t3.py <results root> : per model (bf16 + r985 files found anywhere below root) per task:
base acc, worst base per-class acc, margin stats of base-correct (teacher-forced min-token top1-top2 gap on gold), retention (real gen) + Wilson lo,
worst pooled kind retention (n>=8), worst src x kind. Writes markdown to stdout and JSON summary to T3/summary_<model>.json"""
import json, glob, math, collections, sys, os, statistics
def wl(k, n, z=1.96):
    if n == 0: return 0.0
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); r = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)); return (c - r) / d
root = sys.argv[1]; files = glob.glob(root + "/**/ct_*.json", recursive=True)
by = collections.defaultdict(dict)
for f in files:
    b = os.path.basename(f)[3:-5]; tag, cfg = b.rsplit("_t3_", 1); by[tag][cfg] = f
summ = {}
for tag, cf in sorted(by.items()):
    if "bf16" not in cf: continue
    B = {r["id"]: r for r in json.load(open(cf["bf16"]))["rows"]}
    tasks = sorted(set(r["task"] for r in B.values()))
    for cfg, qf in sorted(cf.items()):
        if cfg == "bf16": continue
        d = json.load(open(qf)); Q = {r["id"]: r for r in d["rows"]}
        print(f"\n### {tag} {cfg} ratio={d['ratio']*100:.3f}% body_bpw={d['body_bpw']:.2f}\n")
        print("| task | n | base acc | worst class base acc | margin>=6 | margin>=10 | retention [Wilson lo] | worst kind ret (n) | worst src x kind (n) |\n|---|---|---|---|---|---|---|---|---|")
        for t in tasks:
            ids = [i for i, r in B.items() if r["task"] == t]; bc = [i for i in ids if B[i]["gen_correct"]]
            kinds = collections.defaultdict(lambda: [0, 0])
            for i in ids: kinds[B[i]["kind"]][1] += 1; kinds[B[i]["kind"]][0] += B[i]["gen_correct"]
            wk = min(a / b for a, b in kinds.values() if b >= 8) if any(b >= 8 for a, b in kinds.values()) else 0
            mm = [B[i]["m_min"] for i in bc]
            k = sum(Q[i]["gen_correct"] for i in bc)
            kd = collections.defaultdict(lambda: [0, 0]); cell = collections.defaultdict(lambda: [0, 0])
            for i in bc:
                r = B[i]; c = kd[r["kind"]]; c[1] += 1; c[0] += Q[i]["gen_correct"]; c = cell[(r["src"], r["kind"])]; c[1] += 1; c[0] += Q[i]["gen_correct"]
            def worst(D):
                v = [(a / b, key, b) for key, (a, b) in D.items() if b >= 8]; return min(v) if v else (0, "-", 0)
            w1 = worst(kd); w2 = worst(cell)
            ret = k / max(1, len(bc))
            summ[(tag, cfg, t)] = dict(n=len(ids), base=len(bc) / len(ids), worstclass=wk, m6=sum(m >= 6 for m in mm) / max(1, len(mm)), m10=sum(m >= 10 for m in mm) / max(1, len(mm)), ret=ret, lo=wl(k, len(bc)), wkind=w1[0], wcell=w2[0], nbc=len(bc))
            print(f"| {t} | {len(ids)} | {len(bc)/len(ids):.3f} | {wk:.2f} | {summ[(tag,cfg,t)]['m6']:.2f} | {summ[(tag,cfg,t)]['m10']:.2f} | {ret:.3f} [{wl(k,len(bc)):.3f}] | {w1[0]:.2f} {w1[1]} ({w1[2]}) | {w2[0]:.2f} {w2[1][0]}/{w2[1][1]} ({w2[2]}) |")
json.dump({"|".join(k): v for k, v in summ.items()}, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "summary_t3.json"), "w"), indent=1)
