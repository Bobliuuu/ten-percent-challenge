"""python analyze_p.py <results root>: per-task-calibrated quantized runs (ct_<model>_t3p<TASK>_r985.json) joined by item id to bf16 base rows
(any ct_<model>_t3*_bf16.json). Prints per (model, task): base acc, retention [Wilson lo], worst kind retention, worst src x kind, margin stats."""
import json, glob, math, collections, sys, os, re
def wl(k, n, z=1.96):
    if n == 0: return 0.0
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); r = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)); return (c - r) / d
root = sys.argv[1]; files = glob.glob(root + "/**/ct_*.json", recursive=True)
base = collections.defaultdict(dict); quant = []
for f in files:
    b = os.path.basename(f)[3:-5]
    m = re.match(r"(.+?)_t3(?:r[23])?_bf16$", b)
    if m:
        for r in json.load(open(f))["rows"]: base[m.group(1)][r["id"]] = r
        continue
    m = re.match(r"(.+?)_t3p([A-Za-z0-9]+)_(r\d+)$", b)
    if m: quant.append((m.group(1), m.group(2), m.group(3), f))
out = {}
print("| model | task | classes | base acc | base-correct | margin>=6 | retention [Wilson lo] | worst kind (n) | worst src x kind (n) | ratio | body bpw |\n|---|---|---|---|---|---|---|---|---|---|---|")
for model, task, cfg, f in sorted(quant):
    d = json.load(open(f)); Q = {r["id"]: r for r in d["rows"]}; B = base[model]
    ids = [i for i in Q if i in B]; bc = [i for i in ids if B[i]["gen_correct"]]
    k = sum(Q[i]["gen_correct"] for i in bc)
    kd = collections.defaultdict(lambda: [0, 0]); cell = collections.defaultdict(lambda: [0, 0])
    for i in bc:
        r = B[i]; c = kd[r["kind"]]; c[1] += 1; c[0] += Q[i]["gen_correct"]; c = cell[(r["src"], r["kind"])]; c[1] += 1; c[0] += Q[i]["gen_correct"]
    def worst(D):
        v = [(a / b, key, b) for key, (a, b) in D.items() if b >= 8]; return min(v) if v else (0, "-", 0)
    w1 = worst(kd); w2 = worst(cell); mm = [B[i]["m_min"] for i in bc]
    out[f"{model}|{task}"] = dict(base=len(bc) / len(ids), n=len(ids), nbc=len(bc), ret=k / max(1, len(bc)), lo=wl(k, len(bc)), wkind=w1[0], wcell=w2[0], ratio=d["ratio"], bpw=d["body_bpw"], kinds={a: (x[0], x[1]) for a, x in kd.items()}, cells={f"{a[0]}/{a[1]}": (x[0], x[1]) for a, x in cell.items()})
    print(f"| {model} | {task} | {len(kd)} | {len(bc)/len(ids):.3f} | {len(bc)} | {sum(m>=6 for m in mm)/max(1,len(mm)):.2f} | {k/max(1,len(bc)):.3f} [{wl(k,len(bc)):.3f}] | {w1[0]:.2f} {w1[1]} ({w1[2]}) | {w2[0]:.2f} {w2[1][0]}/{w2[1][1]} ({w2[2]}) | {d['ratio']*100:.2f}% | {d['body_bpw']:.2f} |")
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "summary_p.json"), "w"), indent=1)
