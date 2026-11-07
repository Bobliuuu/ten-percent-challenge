"""Reads results/ct_<model>_<cfg>.json (from dense_c.py ctasks runs) -> tables + plots. Dev-only probes."""
import json, sys, glob, math, os, collections
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
def wilson(k, n, z=1.96):
    if n == 0: return (0, 0)
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)
model = sys.argv[1]; R = {}
for f in glob.glob(f"results/ct_{model}_*.json"):
    n = os.path.basename(f)[len(f"ct_{model}_"):-5]; R[n] = json.load(open(f))
base = {r["id"]: r for r in R["bf16"]["rows"]}
CF = [c for c in ["t096", "t15", "t20", "t30"] if c in R]
meta = {c: (R[c].get("ratio"), R[c].get("body_bpw")) for c in CF}
tasks = sorted(set(r["task"] for r in base.values()))
out = [f"## {model}: base accuracy, margin and item retention (dev probes, 300 S1 + 300 S3 per task)\n",
       "ratio/bpw: " + ", ".join(f"{c}: {meta[c][0]*100:.2f}% of fp16, body {meta[c][1]:.2f} bpw" for c in CF) + "\n",
       "| task | role | n | base acc | median |margin| (logit) | frac margin<2 | " + " | ".join(f"ret {c}" for c in CF) + " |", "|---|---|---|---|---|---|" + "---|" * len(CF)]
curves = {}
for t in tasks:
    for role in ["S1", "S3", "ALL"]:
        ids = [i for i, r in base.items() if r["task"] == t and (role == "ALL" or r["role"] == role)]
        bc = [i for i in ids if base[i]["correct"]]
        if not ids: continue
        mm = np.array([base[i]["m_first"] for i in bc]) if bc else np.array([0.0])
        cells = []
        for c in CF:
            cr = {r["id"]: r for r in R[c]["rows"]}; k = sum(cr[i]["correct"] for i in bc); lo, hi = wilson(k, len(bc))
            cells.append(f"{k/max(1,len(bc)):.3f} [{lo:.3f},{hi:.3f}]"); 
            if role == "ALL": curves.setdefault(t, []).append((meta[c][1], k / max(1, len(bc)), lo, hi, c))
        out.append(f"| {t} | {role} | {len(ids)} | {len(bc)/len(ids):.3f} | {np.median(mm):.2f} | {(mm<2).mean():.3f} | " + " | ".join(cells) + " |")
open(f"results/table_{model}.md", "w").write("\n".join(out) + "\n"); print("\n".join(out))
fig, ax = plt.subplots(figsize=(7, 4.5))
for t, v in curves.items():
    v.sort(); ax.errorbar([a[0] for a in v], [a[1] for a in v], yerr=[[a[1]-a[2] for a in v], [a[3]-a[1] for a in v]], marker="o", label=t, capsize=2)
ax.axhline(.98, ls="--", c="k", lw=.8); ax.set_xlabel("body bits per weight (GPTQ+rot+ECSQ, dense_c)"); ax.set_ylabel("item retention  P(correct|base correct)"); ax.set_title(f"{model}"); ax.legend(fontsize=7); ax.set_ylim(0, 1.02)
plt.tight_layout(); plt.savefig(f"results/retention_{model}.png", dpi=130)
