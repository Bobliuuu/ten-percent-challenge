import torch, json, sys, math
sys.path.insert(0, "/opt/dlami/nvme/work/l2/code")
from safetensors import safe_open
snap = sys.argv[1]; layers = [int(x) for x in sys.argv[2].split(",")]
wm = json.load(open(f"{snap}/model.safetensors.index.json"))["weight_map"]; files = {}
def get(n):
    f = wm[n]
    if f not in files: files[f] = safe_open(f"{snap}/{f}", "pt", device="cpu")
    return files[f].get_tensor(n)
dev = torch.device("cuda:0")
for bits in (2, 3, 4, 5):
    tot = 0; H = 0
    for l in layers:
        for e in range(0, 128, 8):
            for pn in ("gate_proj", "up_proj", "down_proj"):
                w = get(f"model.layers.{l}.mlp.experts.{e}.{pn}.weight").to(dev).float().reshape(-1, 128)
                qm = 2 ** (bits - 1) - 1; s = w.abs().amax(1, keepdim=True).clamp(min=1e-8) / qm
                q = (w / s).round().clamp(-qm - 1, qm).long().flatten() + qm + 1
                p = torch.bincount(q, minlength=2 ** bits).double(); p /= p.sum()
                H += -(p[p > 0] * p[p > 0].log2()).sum().item() * q.numel(); tot += q.numel()
    print(f"RTN{bits}: entropy {H/tot:.3f} bits/weight (+0.125 scale) -> total {H/tot+0.125:.3f} vs nominal {bits+0.125}", flush=True)
