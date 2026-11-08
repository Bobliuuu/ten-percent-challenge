"""Unfold Ternary-Bonsai-2-27B MLX pack into plain HF bf16 safetensors (text; vision copied from Bonsai-27B-unpacked). Streams shards (62GB RAM box)."""
import os, json, sys, shutil
import torch, numpy as np, scipy.linalg
os.environ["HF_HOME"] = "/home/ubuntu/work/hf"
from huggingface_hub import snapshot_download, hf_hub_download
from safetensors import safe_open
from safetensors.torch import save_file
repo = "prism-ml/Ternary-Bonsai-2-27B-mlx-2bit"; out = "/home/ubuntu/work/tb2_hf"; parent = "Qwen/Qwen3.8-27B"
os.makedirs(out, exist_ok=True)
H = json.load(open("/home/ubuntu/work/hadamard.json")); folded = set(H["prism.hadamard.weight_names"]) | set(H["prism.hadamard.inverse_weight_names"])
sw, sv = H["prism.hadamard.sign_widths"], H["prism.hadamard.sign_values"]; signs = {}; o = 0
for w in sw: signs[w] = torch.tensor(sv[o:o + w], dtype=torch.float32, device="cuda"); o += w
Hm = torch.tensor(scipy.linalg.hadamard(1024), dtype=torch.float32, device="cuda") / 32.0
src = snapshot_download(repo, allow_patterns=["model.safetensors", "*.json", "*.jinja"])
f = safe_open(src + "/model.safetensors", "pt"); keys = list(f.keys())
par = snapshot_download(parent, allow_patterns=["*.json"]); pidx = json.load(open(par + "/model.safetensors.index.json"))["weight_map"]
def hfname(k):
    if k.startswith("language_model.lm_head"): return k.replace("language_model.", "")
    return k.replace("language_model.model.", "model.language_model.")
names = sorted(set(k.rsplit(".", 1)[0] for k in keys if k.endswith(".scales")))
cur = {}; size = 0; shard_i = 0; wm = {}; report = {"norm_stats": {}}
def flush():
    global cur, size, shard_i
    if not cur: return
    shard_i += 1; fn = f"model-{shard_i:05d}.safetensors"; save_file(cur, out + "/" + fn, metadata={"format": "pt"})
    for k in cur: wm[k] = fn
    cur = {}; size = 0
def put(k, t):
    global size
    cur[k] = t.contiguous(); size += t.numel() * t.element_size()
    if size > 4.5e9: flush()
for base in names:
    w = f.get_tensor(base + ".weight").to("cuda"); s = f.get_tensor(base + ".scales").to("cuda").float(); b = f.get_tensor(base + ".biases").to("cuda").float()
    w = w.view(torch.int32) if w.dtype != torch.int32 else w
    rows = w.shape[0]; sh = torch.arange(0, 32, 2, device="cuda", dtype=torch.int32)
    codes = ((w[:, :, None] >> sh) & 3).reshape(rows, -1).float(); cols = codes.shape[1]
    W = (codes.reshape(rows, -1, 128) * s[..., None] + b[..., None]).reshape(rows, cols)
    wn = base + ".weight"
    if wn in folded: W = (W.reshape(rows, cols // 1024, 1024) @ Hm).reshape(rows, cols) * signs[cols]
    put(hfname(wn), W.to(torch.bfloat16).cpu()); del w, codes, W
small = {}
for k in keys:
    if k.endswith((".scales", ".biases")): continue
    if k.rsplit(".", 1)[0] in names and k.endswith(".weight"): continue
    t = f.get_tensor(k)
    if t.dtype in (torch.uint32, torch.int32): continue
    hk = hfname(k)
    if hk.endswith("conv1d.weight") and t.ndim == 3 and t.shape[-1] == 1: t = t.permute(0, 2, 1).contiguous()
    if hk.endswith(("input_layernorm.weight", "post_attention_layernorm.weight", "language_model.norm.weight", "q_norm.weight", "k_norm.weight")): t = (t.float() - 1.0)
    small[hk] = t
for nm in ["model.language_model.layers.0.input_layernorm.weight", "model.language_model.norm.weight", "model.language_model.layers.3.self_attn.q_norm.weight", "model.language_model.layers.0.linear_attn.norm.weight", "model.language_model.layers.0.linear_attn.A_log", "model.language_model.layers.0.linear_attn.dt_bias", "model.language_model.layers.0.linear_attn.conv1d.weight", "model.language_model.layers.0.linear_attn.in_proj_a.weight"]:
    sh_ = pidx.get(nm)
    if sh_ and nm in small:
        pt = safe_open(hf_hub_download(parent, sh_), "pt").get_tensor(nm).float(); mine = small[nm].float()
        report["norm_stats"][nm] = dict(parent_mean=float(pt.mean()), mine_mean=float(mine.mean()), maxdiff=float((pt - mine).abs().max()), maxdiff_minus1=float((pt - (mine - 1)).abs().max()), pm=float(pt.abs().mean()), shape=list(pt.shape))
print("NORMCHK", json.dumps(report), flush=True)
for k, v in small.items(): put(k, v.to(torch.bfloat16) if v.dtype == torch.float16 else v)
bsrc = snapshot_download("prism-ml/Bonsai-27B-unpacked", allow_patterns=["*.json"])
bidx = json.load(open(bsrc + "/model.safetensors.index.json"))["weight_map"]
for sh_ in sorted(set(v for k, v in bidx.items() if k.startswith("model.visual"))):
    with safe_open(hf_hub_download("prism-ml/Bonsai-27B-unpacked", sh_), "pt") as g:
        for k in g.keys():
            if k.startswith("model.visual"): put(k, g.get_tensor(k))
flush()
# rename shards with total count in index (names fine as is)
json.dump({"metadata": {}, "weight_map": wm}, open(out + "/model.safetensors.index.json", "w"))
psrc = snapshot_download(parent, allow_patterns=["*.json", "*.jinja", "*.txt"])
for fn in os.listdir(psrc):
    if fn != "model.safetensors.index.json" and os.path.isfile(psrc + "/" + fn): shutil.copy(psrc + "/" + fn, out + "/" + fn)
ref = {}
for sh_ in sorted(set(pidx.values())):
    with safe_open(hf_hub_download(parent, sh_), "pt") as g:
        for k in g.keys():
            if k.startswith(("model.language_model", "lm_head")): ref[k] = list(g.get_slice(k).get_shape())
mine_shapes = {}
for fn in sorted(set(wm.values())):
    with safe_open(out + "/" + fn, "pt") as g:
        for k in g.keys(): mine_shapes[k] = list(g.get_slice(k).get_shape())
print("SHAPE_MISMATCH", [(k, mine_shapes.get(k), ref[k]) for k in ref if mine_shapes.get(k) != ref[k]][:20], "extra", [k for k in mine_shapes if k not in ref and not k.startswith("model.visual")][:10], flush=True)
print("DONE", flush=True)
