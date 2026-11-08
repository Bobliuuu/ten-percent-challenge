"""Reads results/<model>_r1.json (dict cfgname -> ct rows). Per task: base acc, margin stats, retention@cfg (overall, per-class min, per-source), Wilson."""
import json, sys, math, collections, numpy as np
def wil(k, n, z=1.96):
    if n == 0: return (0., 0.)
    p = k / n; d = 1 + z*z/n; c = p + z*z/(2*n); h = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n)); return ((c-h)/d, (c+h)/d)
def load(model, tag="r1"): return json.load(open(f"results/{model}_{tag}.json"))
def table(model, cfgs=("r096", "r100", "r1125", "r125")):
    R = load(model); base = {r["id"]: r for r in R["bf16"]["rows"]}; out = []; summ = {}
    meta = {c: (R[c]["ratio"], R[c]["body_bpw"]) for c in cfgs if c in R}
    out.append(f"### {model}: " + ", ".join(f"{c}: {m[0]*100:.2f}% of fp16 / body {m[1]:.2f} bpw" for c, m in meta.items()))
    out.append("| task | n | classes | base acc | base acc S1/S3 | med margin | %margin<2 | " + " | ".join(f"ret {c}" for c in meta) + " | " + " | ".join(f"min-class {c}" for c in meta) + " |")
    out.append("|---|---|---|---|---|---|---|" + "---|" * (2 * len(meta)))
    for t in sorted(set(r["task"] for r in base.values())):
        ids = [i for i, r in base.items() if r["task"] == t]; bc = [i for i in ids if base[i]["correct"]]
        a = lambda role: np.mean([base[i]["correct"] for i in ids if base[i]["role"] == role])
        mm = np.array([base[i]["m_min"] for i in bc]); cls = sorted(set(base[i]["kind"] for i in ids))
        rets, mins = [], []; summ[t] = dict(base=len(bc)/len(ids), med=float(np.median(mm)), lt2=float((mm < 2).mean()), ncls=len(cls))
        for c in meta:
            cr = {r["id"]: r for r in R[c]["rows"]}; k = sum(cr[i]["correct"] for i in bc); lo, hi = wil(k, len(bc))
            per = {}
            for kd in cls:
                b2 = [i for i in bc if base[i]["kind"] == kd]
                if len(b2) >= 8: per[kd] = (sum(cr[i]["correct"] for i in b2)/len(b2), len(b2))
            mk = min(per, key=lambda x: per[x][0]) if per else None
            rets.append(f"{k/len(bc):.3f} [{lo:.3f}]"); mins.append(f"{per[mk][0]:.2f} ({mk},n={per[mk][1]})" if mk else "-")
            summ[t][c] = (k/len(bc), lo, per[mk][0] if mk else None)
        out.append(f"| {t} | {len(ids)} | {len(cls)} | {len(bc)/len(ids):.3f} | {a('S1'):.2f}/{a('S3'):.2f} | {np.median(mm):.1f} | {(mm<2).mean():.3f} | " + " | ".join(rets) + " | " + " | ".join(mins) + " |")
    return "\n".join(out), summ
if __name__ == "__main__":
    for m in sys.argv[1:]:
        s, d = table(m); print(s); open(f"results/table_{m}.md", "w").write(s + "\n"); json.dump(d, open(f"results/summ_{m}.json", "w"), indent=1)
