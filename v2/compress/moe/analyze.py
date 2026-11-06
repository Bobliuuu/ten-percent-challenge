"""Stage-0 ranking (successive halving), stride-additivity extrapolation, KL->retention fit. Reads results/vol/*.json (collect.py first)."""
import json, glob, numpy as np
R = {json.load(open(f))["name"]: json.load(open(f)) for f in glob.glob("results/vol/*.json") if "ladder" not in f}
out = []
# 1) stage-0 ranking: stage0_em16_A (64 prompts) then top1/KL
rk = sorted([r for r in R.values() if "stage0_em16_A" in r and not r["name"].startswith(("stride", "smoke"))], key=lambda r: (-r["stage0_em16_A"], r.get("kl_A", 9)))
out.append("## Stage-0 ranking (em16 on 64 prompts, ties by KL); bottom half = would be killed")
for i, r in enumerate(rk): out.append(f"{i+1}. {r['name']}: em16={r['stage0_em16_A']:.3f} top1={r['top1_A']:.3f} kl={r['kl_A']:.4f} bytes%={100*r['ratio']:.2f} {'KILL' if i >= len(rk)/2 else 'keep'}")
# 2) stride additivity: sum of KL over the 4 offsets approximates full-model KL (first-order additive)
for k in (64, 32):
    xs = [R.get(f"stride4_k{k}_o{o}") for o in range(4)]
    if all(xs):
        pred = sum(x["kl_A"] for x in xs); full = R.get(f"fillA_k{k}")
        out.append(f"\n## Additivity k={k}: sum KL over 4 stride offsets = {pred:.4f}; per-offset {[round(x['kl_A'],4) for x in xs]}; actual full-model KL = {full['kl_A'] if full else 'n/a'}")
# 3) KL -> retention fit (logistic-ish: ret = exp(-a*KL)) on all configs with item-retention
pts = []
for r in R.values():
    ret = r.get("retention_A", {})
    for c in ("agnews", "sst2", "json"):
        if c in ret and ret[c]["retention"] is not None and r.get("kl_A") is not None: pts.append((r["kl_A"], ret[c]["retention"], c))
if len(pts) >= 4:
    kl = np.array([p[0] for p in pts]); y = np.array([p[1] for p in pts]); a = -np.sum(kl * np.log(np.clip(y, 1e-3, 1))) / np.sum(kl * kl)
    out.append(f"\n## KL->retention fit: ret = exp(-a*KL), a={a:.2f} (n={len(pts)} cat-config points); KL for 98% = {-np.log(.98)/a:.5f}")
open("results/analysis.md", "w").write("\n".join(out)); print("\n".join(out))
