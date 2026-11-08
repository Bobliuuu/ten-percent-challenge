"""KL-sensitivity bit allocation for qpush plans.
CLI: python alloc.py results/sens7b_k13.json TARGET_BODY_BPW [a=2.5] -> plan JSON (list of [regex, K])
Library (X03 / A1): plan_from_sens(sens_json, target_bpw, a, grid=None) -> {"L<li>.<group>": K}
Model: KL_g(b) = s_g * 2^(-a (b - b0)), b0 = log2(sens_K)/8; minimise sum KL_g s.t. sum n_g b_g = B*N; K_g = 2^(8 b_g) rounded (optionally snapped to a K grid), clipped to [2^10, 2^16]."""
import json, sys, math, numpy as np
names = {"q_proj": ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"], "o_proj": ["self_attn.o_proj"], "gate_proj": ["mlp.gate_proj", "mlp.up_proj"], "down_proj": ["mlp.down_proj"]}
def plan_from_sens(r, target, a=2.5, grid=None, lo=10 / 8, hi=16 / 8):
    off = r["sens_kl_offset"]; b0 = math.log2(r["sens_K"]) / 8
    keys = list(r["sens"].keys()); s = np.array([max(r["sens"][k]["kl"] - off, 1e-6) for k in keys]); n = np.array([r["sens"][k]["n"] for k in keys], float)
    def bits(lam): return np.clip(b0 + np.log2(a * math.log(2) * s / (lam * n)) / a, lo, hi)
    L, H = 1e-12, 1e3
    for _ in range(200):
        lam = math.sqrt(L * H); bb = bits(lam)
        if (bb * n).sum() / n.sum() > target: L = lam
        else: H = lam
    bb = bits(H); K = np.clip(np.round(2 ** (8 * bb)), 1024, 65536).astype(int)
    if grid is not None:
        g = np.array(grid); K = np.array([g[np.argmin(np.abs(np.log2(g) - np.log2(k)))] for k in K])
    return {k: int(kk) for k, kk in zip(keys, K)}, (s, n, b0, keys)
if __name__ == "__main__":
    r = json.load(open(sys.argv[1])); target = float(sys.argv[2]); a = float(sys.argv[3]) if len(sys.argv) > 3 else 2.5
    pl, (s, n, b0, keys) = plan_from_sens(r, target, a); K = np.array([pl[k] for k in keys])
    kl = lambda b: (s * 2 ** (-a * (b - b0))).sum()
    print("uniform b0 pred KL %.4f at %.3f bpw ; alloc pred KL %.4f at %.3f bpw" % (kl(np.full(len(s), b0)), b0, kl(np.log2(K) / 8), ((np.log2(K) / 8) * n).sum() / n.sum()), file=sys.stderr)
    plan = []
    for k, kk in zip(keys, K):
        li, g = k.split(".")[0][1:], k.split(".")[1]
        for nm in names[g]: plan.append(["^L%s\\.%s$" % (li, nm.replace(".", "\\.")), int(kk)])
    json.dump(plan, open(sys.argv[1].replace(".json", f"_plan_{target}.json"), "w"))
    import collections
    byg = collections.defaultdict(list)
    for k, kk in zip(keys, K): byg[k.split(".")[1]].append(math.log2(kk) / 8)
    print({g: round(float(np.mean(v)), 3) for g, v in byg.items()}, file=sys.stderr)
    print(json.dumps(plan))
