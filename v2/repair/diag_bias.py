"""Bias-flip diagnosis from existing per-item margins (base vs scalar GPTQ 9.86%): is the loss a systematic logit shift per class?
usage: python diag_bias.py BASE_ct.json CAND_ct.json"""
import json, sys, collections, numpy as np
b = {r["id"]: r for r in json.load(open(sys.argv[1]))["rows"]}; c = {r["id"]: r for r in json.load(open(sys.argv[2]))["rows"]}
cells = collections.defaultdict(list)
for i, rb in b.items():
    rc = c[i]; cells[(rb["task"], rb["src"] + "_" + rb["kind"])].append((rb["m_first"], rc["m_first"], rb["correct"], rc["correct"]))
print("cell n  base_acc cand_acc | mean dm  std dm | ret | frac base-correct w/ m<5 | oracle-shift flips")
for k, v in sorted(cells.items()):
    a = np.array(v, float); dm = a[:, 1] - a[:, 0]; bc = a[:, 2] == 1
    ret = a[bc, 3].mean(); small = (a[bc, 0] < 5).mean()
    # oracle: shift cell margins by -mean dm (class-constant logit bias): how many base-correct kept
    kept = ((a[bc, 1] - dm.mean()) > 0).mean()
    print(f"{k[0]:5s} {k[1]:22s} {len(v):5d} {a[:,2].mean():.3f} {a[:,3].mean():.3f} | {dm.mean():+7.2f} {dm.std():6.2f} | {ret:.3f} | {small:.3f} | {kept:.3f}")
