"""Scale statistics of Ternary-Bonsai-2-27B MLX pack: entropy of log-quantised group scales. CPU modal."""
import modal, os, json
app = modal.App("tenpct-native2-scales")
img = modal.Image.debian_slim(python_version="3.11").pip_install("torch==2.5.1", "numpy", "safetensors", "huggingface_hub", "hf_transfer").env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
vol = modal.Volume.from_name("tenpct-native2", create_if_missing=True)
@app.function(image=img, cpu=16, memory=65536, timeout=7200, volumes={"/vol": vol})
def run():
    import numpy as np, torch, math
    os.environ["HF_HOME"] = "/vol/hf"
    from huggingface_hub import snapshot_download
    from safetensors import safe_open
    src = snapshot_download("prism-ml/Ternary-Bonsai-2-27B-mlx-2bit", allow_patterns=["model.safetensors"])
    f = safe_open(src + "/model.safetensors", "pt"); R = {}
    Ks = [1, 2, 4, 8, 16]; hist = {K: {} for K in Ks}; n = 0; spread = []; tens = []
    rowcnt = 0
    for k in f.keys():
        if not k.endswith(".scales"): continue
        s = f.get_tensor(k).float().abs().clamp_min(1e-12); L = torch.log2(s)  # [rows, ng]
        n += s.numel(); med = L.median(dim=1, keepdim=True).values; rowcnt += s.shape[0]
        spread.append(float((L - med).abs().mean()))
        for K in Ks:
            q = torch.round((L - med) * K).to(torch.int32).flatten().numpy()
            u, c = np.unique(q, return_counts=True); h = hist[K]
            for a, b in zip(u.tolist(), c.tolist()): h[a] = h.get(a, 0) + b
        if len(tens) < 6 or "embed" in k: tens.append((k, list(s.shape), float(L.std()), float((L - med).std())))
    R["n_scales"] = n; R["rows"] = rowcnt; R["mean_abs_dev_log2_from_row_median"] = float(np.mean(spread))
    for K in Ks:
        h = hist[K]; tot = sum(h.values()); H = -sum(c / tot * math.log2(c / tot) for c in h.values()); R[f"K{K}"] = dict(entropy_bits=H, levels=len(h), maxrelerr_pct=100 * (2 ** (0.5 / K) - 1), min=min(h), max=max(h))
    R["tensors"] = tens; json.dump(R, open("/vol/scale_stats.json", "w")); vol.commit(); print("RESULT_JSON", json.dumps(R))
@app.local_entrypoint()
def main(): run.remote()
