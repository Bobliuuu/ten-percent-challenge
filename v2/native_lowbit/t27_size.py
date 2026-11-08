"""Ternary-Bonsai-2-27B size probe: exact trit histogram from the MLX 2-bit pack (U32 codes) -> entropy-coded bytes. CPU Modal."""
import modal, os, json
app = modal.App("tenpct-native-t27size")
img = modal.Image.debian_slim(python_version="3.11").pip_install("torch==2.5.1", "numpy", "safetensors", "huggingface_hub")
vol = modal.Volume.from_name("tenpct-native", create_if_missing=True)
@app.function(image=img, cpu=16, memory=65536, timeout=7200, volumes={"/vol": vol})
def run():
    import math, numpy as np, torch, collections
    os.environ["HF_HOME"] = "/vol/hf"
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open
    p = hf_hub_download("prism-ml/Ternary-Bonsai-2-27B-mlx-2bit", "model.safetensors")
    f = safe_open(p, "pt"); tot = np.zeros(4); rowbits = 0.0; gbits = 0.0; n = 0; per = []; sh = torch.arange(0, 32, 2)
    names = list(f.keys()); smallinfo = collections.Counter(); ex = {}
    for k in names:
        t = f.get_tensor(k)
        if t.dtype == torch.int32 or str(t.dtype) == "torch.uint32":
            w = t.view(torch.int32) if t.dtype != torch.int32 else t
            c = torch.stack([((w[:, :, None] >> sh) & 3).reshape(w.shape[0], -1)] ).squeeze(0)  # [rows, cols]
            cnt = torch.stack([(c == v).sum(1) for v in range(4)], 1).double().numpy()   # per row
            g = cnt.sum(0); tot += g; N = g.sum(); n += N
            p_ = g / N; H = -sum(x * math.log2(x) for x in p_ if x > 0); gbits += H * N
            pr = cnt / cnt.sum(1, keepdims=True); Hr = -np.where(pr > 0, pr * np.log2(np.where(pr > 0, pr, 1)), 0).sum(1); rowbits += float((Hr * cnt.sum(1)).sum()) + 48 * cnt.shape[0]
            if len(per) < 6 or "embed" in k or "lm_head" in k: per.append((k, list(t.shape), (g / N).round(4).tolist(), round(H, 4)))
        else:
            smallinfo[(str(t.dtype), k.split(".")[-1])] += t.numel() * t.element_size()
            if t.dtype == torch.float16 and k.endswith("weight") and len(ex) < 12: ex[k] = list(t.shape)
    # scale/bias analysis: is bias == -scale (symmetric ternary)?
    chk = []
    for k in names:
        if k.endswith(".scales"):
            s = f.get_tensor(k); b = f.get_tensor(k[:-7] + ".biases"); chk.append((k, float((b.float() + s.float()).abs().max()), float(s.float().abs().mean()))); 
            if len(chk) >= 6: break
    R = dict(trit_params=float(n), code_hist=(tot / n).tolist(), global_entropy_bpw=gbits / n, rowwise_entropy_bpw=rowbits / n, per_tensor=per, small=[(list(k), v) for k, v in smallinfo.items()],
             f16_weight_examples=ex, scale_bias_check=chk, nonternary_names=[k for k in names if k.endswith("weight") and f.get_tensor(k).dtype == torch.float16][:60])
    json.dump(R, open("/vol/t27_size.json", "w")); vol.commit(); print("RESULT_JSON", json.dumps(R))
