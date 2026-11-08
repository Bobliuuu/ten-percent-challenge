"""Per-tensor-group TCQ rate plan from KL sensitivities (entry32b/results/e32_sens.json measured at 8-d VQ K=8192, b0=13/8 bpw).
Model KL_g(b) = s_g * 2^(-a (b-b0)) (a=2.5 as in quality/alloc.py). Choose kb in {2,3,4} (V=2 => 1.0/1.5/2.0 bpw) per group minimising sum KL
s.t. mean bpw = target. Greedy Lagrangian on discrete options. usage: python alloc_tcq.py e32_sens.json TARGET_BPW > plan.json"""
import json, sys, math, numpy as np
r = json.load(open(sys.argv[1])); target = float(sys.argv[2]); a = 2.5
off = r["sens_kl_offset"]; b0 = math.log2(r["sens_K"]) / 8
names = {"q_proj": ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"], "o_proj": ["self_attn.o_proj"], "gate_proj": ["mlp.gate_proj", "mlp.up_proj"], "down_proj": ["mlp.down_proj"]}
keys = list(r["sens"].keys()); s = np.array([max(r["sens"][k]["kl"] - off, 1e-6) for k in keys]); n = np.array([r["sens"][k]["n"] for k in keys], float)
opts = np.array([1.0, 1.5, 2.0]); kbs = [2, 3, 4]
cost = s[:, None] * 2.0 ** (-a * (opts[None] - b0))  # [G,3]
def pick(lam):
    tot = cost + lam * n[:, None] * opts[None] / n.sum(); j = tot.argmin(1); return j, (n * opts[j]).sum() / n.sum()
lo, hi = 1e-12, 1e3
for _ in range(200):
    lam = math.sqrt(lo * hi); j, b = pick(lam)
    if b > target: lo = lam
    else: hi = lam
j, b = pick(hi)
plan = []
for k, jj in zip(keys, j):
    li, g = k.split(".")[0][1:], k.split(".")[1]
    for nm in names[g]: plan.append(["^L%s\\.%s$" % (li, nm.replace(".", "\\.")), kbs[jj]])
print(json.dumps(plan))
import collections
print("mean bpw", b, "pred KL", cost[np.arange(len(j)), j].sum(), "vs uniform1.5", cost[:, 1].sum(), collections.Counter(int(kbs[x]) for x in j), file=sys.stderr)
