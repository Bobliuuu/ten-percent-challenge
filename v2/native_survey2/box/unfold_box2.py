"""Unfold Ternary-Bonsai-2-27B MLX pack into plain HF bf16 safetensors (text; vision copied from Bonsai-27B-unpacked). Streams shards (62GB RAM box)."""
import os, json, sys, shutil
import torch, numpy as np, scipy.linalg
DEV = os.environ.get("DEV", "cuda")
os.environ["HF_HOME"] = "/home/ubuntu/work/hf"
from huggingface_hub import snapshot_download, hf_hub_download
from safetensors import safe_open
from safetensors.torch import save_file
K = int(sys.argv[1]) if len(sys.argv) > 1 else 0   # scale log2 resolution steps per octave (0 = exact scales)
NV = int(sys.argv[2]) if len(sys.argv) > 2 else 0  # vocab prefix length (0 = no trim)
LAYERS_DROP = [int(x) for x in sys.argv[4].split(",")] if len(sys.argv) > 4 and sys.argv[4] else []
repo = "prism-ml/Ternary-Bonsai-2-27B-mlx-2bit"; out = sys.argv[3] if len(sys.argv) > 3 else "/home/ubuntu/work/tb2_hf"; parent = "Qwen/Qwen3.8-27B"
SPEC0, SPEC1 = 248044, 248077
keep_rows = (list(range(NV)) + list(range(SPEC0, SPEC1))) if NV else None
os.makedirs(out, exist_ok=True)
H = json.load(open("/home/ubuntu/work/hadamard.json")); folded = set(H["prism.hadamard.weight_names"]) | set(H["prism.hadamard.inverse_weight_names"])
sw, sv = H["prism.hadamard.sign_widths"], H["prism.hadamard.sign_values"]; signs = {}; o = 0
for w in sw: signs[w] = torch.tensor(sv[o:o + w], dtype=torch.float32, device=DEV); o += w
Hm = torch.tensor(scipy.linalg.hadamard(1024), dtype=torch.float32, device=DEV) / 32.0
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
    w = f.get_tensor(base + ".weight").to(DEV); s = f.get_tensor(base + ".scales").to(DEV).float(); b = f.get_tensor(base + ".biases").to(DEV).float()
    w = w.view(torch.int32) if w.dtype != torch.int32 else w
    rows = w.shape[0]; sh = torch.arange(0, 32, 2, device=DEV, dtype=torch.int32)
    codes = ((w[:, :, None] >> sh) & 3).reshape(rows, -1).float(); cols = codes.shape[1]
    if K:
        L = torch.log2(s.abs().clamp_min(1e-12)); med = L.median(dim=1, keepdim=True).values.half().float()
        q = torch.round((L - med) * K); s = torch.sign(s) * torch.exp2(med + q / K); b = -s
    W = (codes.reshape(rows, -1, 128) * s[..., None] + b[..., None]).reshape(rows, cols)
    wn = base + ".weight"
    if wn in folded: W = (W.reshape(rows, cols // 1024, 1024) @ Hm).reshape(rows, cols) * signs[cols]
    if keep_rows is not None and (wn.endswith("embed_tokens.weight") or wn.endswith("lm_head.weight")): W = W[torch.tensor(keep_rows, device=DEV)]
    put(hfname(wn), W.to(torch.bfloat16).cpu()); del w, codes, W
small = {}
for k in keys:
    if k.endswith((".scales", ".biases", ".signs")) or k.startswith("vision_tower"): continue
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
    if fn != "model.safetensors.index.json" and os.path.isfile(psrc + "/" + fn):
        if os.path.lexists(out + "/" + fn): os.remove(out + "/" + fn)
        shutil.copyfile(psrc + "/" + fn, out + "/" + fn)

if NV:
    tj = json.load(open(out + "/tokenizer.json")); m = tj["model"]
    m["vocab"] = {t: i for t, i in m["vocab"].items() if i < NV}; kv = m["vocab"]; m["merges"] = [mg for mg in m["merges"] if (lambda a, b: a in kv and b in kv and (a + b) in kv)(*(mg if isinstance(mg, list) else mg.split(" ")))]
    sp = {a["id"]: a for a in tj["added_tokens"]}; mp = {}; na = []
    for j, oid in enumerate(range(SPEC0, SPEC1)):
        a = dict(sp[oid]); a["id"] = NV + j; mp[oid] = NV + j; na.append(a)
    tj["added_tokens"] = na; json.dump(tj, open(out + "/tokenizer.json", "w"), ensure_ascii=False)
    for fn in ("vocab.json", "merges.txt"):
        if os.path.exists(out + "/" + fn): os.remove(out + "/" + fn)
    def remap(o):
        if isinstance(o, dict):
            r = {}
            for k, v in o.items():
                if k.endswith("token_id") and isinstance(v, int) and v in mp: v = mp[v]
                elif k.endswith("token_ids") and isinstance(v, list): v = [mp.get(x, x) for x in v]
                elif k == "eos_token_id" and isinstance(v, list): v = [mp.get(x, x) for x in v]
                elif k == "vocab_size" and isinstance(v, int): v = NV + (SPEC1 - SPEC0)
                else: v = remap(v)
                r[k] = v
            return r
        return o
    for fn in ("config.json", "generation_config.json"):
        c = json.load(open(out + "/" + fn)); json.dump(remap(c), open(out + "/" + fn, "w"), indent=1)
    tc = json.load(open(out + "/tokenizer_config.json"))
    if "added_tokens_decoder" in tc: tc["added_tokens_decoder"] = {str(mp.get(int(k), int(k))): v for k, v in tc["added_tokens_decoder"].items() if int(k) in mp}
    json.dump(tc, open(out + "/tokenizer_config.json", "w"), ensure_ascii=False)
ref = {}
for sh_ in []:
    with safe_open(hf_hub_download(parent, sh_), "pt") as g:
        for k in g.keys():
            if k.startswith(("model.language_model", "lm_head")): ref[k] = list(g.get_slice(k).get_shape())
mine_shapes = {}
for fn in sorted(set(wm.values())):
    with safe_open(out + "/" + fn, "pt") as g:
        for k in g.keys(): mine_shapes[k] = list(g.get_slice(k).get_shape())
print("SHAPE_MISMATCH", [(k, mine_shapes.get(k), ref[k]) for k in ref if mine_shapes.get(k) != ref[k]][:20], "extra", [k for k in mine_shapes if k not in ref and not k.startswith("model.visual")][:10], flush=True)
print("DONE", flush=True)
