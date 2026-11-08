"""Unfold Ternary-Bonsai-2-27B MLX pack (Hadamard-folded, affine 2-bit g128) into plain HF bf16 safetensors (text + vision copied from Bonsai-27B-unpacked).
W_orig = (W_deq @ blockdiag(H1024_normalized)) * signs  (columns), also for embedding rows (inverse). modal run --detach unfold.py"""
import modal, os, json
HERE = os.path.dirname(os.path.abspath(__file__))
app = modal.App("tenpct-native2-unfold")
img = (modal.Image.debian_slim(python_version="3.11").pip_install("torch==2.5.1", "numpy", "scipy", "safetensors", "huggingface_hub", "hf_transfer").env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
       .add_local_file(HERE + "/hadamard.json", "/app/hadamard.json"))
vol = modal.Volume.from_name("tenpct-native2", create_if_missing=True)
@app.function(image=img, gpu="H100", cpu=8, memory=98304, timeout=4*3600, volumes={"/vol": vol})
def run(repo="prism-ml/Ternary-Bonsai-2-27B-mlx-2bit", out="/vol/tb2_hf", parent="Qwen/Qwen3.8-27B"):
    import torch, numpy as np, scipy.linalg, shutil
    os.environ["HF_HOME"] = "/vol/hf"
    from huggingface_hub import snapshot_download, hf_hub_download
    from safetensors import safe_open
    from safetensors.torch import save_file
    os.makedirs(out, exist_ok=True)
    H = json.load(open("/app/hadamard.json")); folded = set(H["prism.hadamard.weight_names"]) | set(H["prism.hadamard.inverse_weight_names"])
    sw, sv = H["prism.hadamard.sign_widths"], H["prism.hadamard.sign_values"]; signs = {}; o = 0
    for w in sw: signs[w] = torch.tensor(sv[o:o + w], dtype=torch.float32, device="cuda"); o += w
    Hm = torch.tensor(scipy.linalg.hadamard(1024), dtype=torch.float32, device="cuda") / 32.0
    src = snapshot_download(repo, allow_patterns=["model.safetensors", "*.json", "*.jinja"])
    f = safe_open(src + "/model.safetensors", "pt"); keys = list(f.keys())
    par = snapshot_download(parent, allow_patterns=["*.json"])
    pidx = json.load(open(par + "/model.safetensors.index.json"))["weight_map"]
    def hfname(k):
        if k.startswith("language_model.lm_head"): return k.replace("language_model.", "")
        return k.replace("language_model.model.", "model.language_model.")
    report = {"norm_stats": {}, "unmatched": []}
    sd = {}; names = set(k.rsplit(".", 1)[0] for k in keys if k.endswith(".scales"))
    for base in sorted(names):
        w = f.get_tensor(base + ".weight").to("cuda"); s = f.get_tensor(base + ".scales").to("cuda").float(); b = f.get_tensor(base + ".biases").to("cuda").float()
        w = w.view(torch.int32) if w.dtype != torch.int32 else w
        rows = w.shape[0]; sh = torch.arange(0, 32, 2, device="cuda", dtype=torch.int32)
        codes = ((w[:, :, None] >> sh) & 3).reshape(rows, -1).float()
        cols = codes.shape[1]
        W = (codes.reshape(rows, -1, 128) * s[..., None] + b[..., None]).reshape(rows, cols)
        wn = base + ".weight"
        if wn in folded:
            W = (W.reshape(rows, cols // 1024, 1024) @ Hm).reshape(rows, cols) * signs[cols]
        sd[hfname(wn)] = W.to(torch.bfloat16).cpu(); del w, codes, W
    for k in keys:
        if k.endswith((".scales", ".biases")): continue
        t = f.get_tensor(k)
        if k.rsplit(".", 1)[0] in names and k.endswith(".weight"): continue
        hk = hfname(k)
        if t.dtype == torch.uint32 or t.dtype == torch.int32: continue
        if hk.endswith("conv1d.weight") and t.ndim == 3 and t.shape[-1] == 1: t = t.transpose(1, 2).contiguous() if False else t.permute(0, 2, 1).contiguous()
        sd[hk] = t
    # norm convention check vs parent for layer0 input_layernorm + final norm
    for nm in ["model.language_model.layers.0.input_layernorm.weight", "model.language_model.norm.weight", "model.language_model.layers.3.self_attn.q_norm.weight", "model.language_model.layers.0.linear_attn.norm.weight"]:
        sh_ = pidx.get(nm)
        if sh_ and nm in sd:
            pf = hf_hub_download(parent, sh_); pt = safe_open(pf, "pt").get_tensor(nm).float()
            mine = sd[nm].float(); report["norm_stats"][nm] = dict(parent_mean=float(pt.mean()), mine_mean=float(mine.mean()), maxdiff=float((pt - mine).abs().max()), maxdiff_plus1=float((pt - (mine - 1)).abs().max()) , shape=list(pt.shape))
    print("NORMCHK", json.dumps(report), flush=True)
    report["n_tensors"] = len(sd)
    # parent comparison of dtype/shape for all text tensors
    ref_shapes = {}
    for sh_ in sorted(set(pidx.values())):
        pf = hf_hub_download(parent, sh_)
        with safe_open(pf, "pt") as g:
            for k in g.keys():
                if k.startswith("model.language_model") or k.startswith("lm_head"): ref_shapes[k] = list(g.get_slice(k).get_shape())
    bad = [(k, list(v.shape), ref_shapes.get(k)) for k, v in sd.items() if ref_shapes.get(k) != list(v.shape)]
    report["shape_mismatch"] = bad[:20]; report["missing_vs_parent"] = [k for k in ref_shapes if k not in sd][:20]; print("SHAPECHK", json.dumps(report), flush=True)
    # vision from Bonsai-27B-unpacked
    bsrc = snapshot_download("prism-ml/Bonsai-27B-unpacked", allow_patterns=["*.json", "*.jinja", "merges.txt"])
    bidx = json.load(open(bsrc + "/model.safetensors.index.json"))["weight_map"]
    vis = {}
    for sh_ in sorted(set(v for k, v in bidx.items() if k.startswith("model.visual"))):
        pf = hf_hub_download("prism-ml/Bonsai-27B-unpacked", sh_)
        with safe_open(pf, "pt") as g:
            for k in g.keys():
                if k.startswith("model.visual"): vis[k] = g.get_tensor(k)
    # shard write
    items = list(sd.items()) + list(vis.items()); shards = []; cur = {}; size = 0
    for k, v in items:
        cur[k] = v.contiguous(); size += v.numel() * v.element_size()
        if size > 4.5e9: shards.append(cur); cur = {}; size = 0
    if cur: shards.append(cur)
    wm = {}
    for i, sdict in enumerate(shards):
        fn = f"model-{i+1:05d}-of-{len(shards):05d}.safetensors"; save_file(sdict, out + "/" + fn, metadata={"format": "pt"})
        for k in sdict: wm[k] = fn
    json.dump({"metadata": {}, "weight_map": wm}, open(out + "/model.safetensors.index.json", "w"))
    psrc = snapshot_download(parent, allow_patterns=["*.json", "*.jinja", "*.txt"])
    for fn in os.listdir(psrc):
        if fn != "model.safetensors.index.json" and os.path.isfile(psrc + "/" + fn): shutil.copy(psrc + "/" + fn, out + "/" + fn)
    vol.commit(); print("DONE", json.dumps(report), flush=True)
@app.local_entrypoint()
def main(): run.remote()
