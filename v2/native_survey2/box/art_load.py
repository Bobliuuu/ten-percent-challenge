"""Shipped loader: artifact dir -> plain HF bf16 safetensors (text LM). usage: art_load.py ART OUT [cpu|cuda]"""
import os, sys, json, lzma, numpy as np, torch, scipy.linalg
from safetensors.torch import save_file
art, out = sys.argv[1], sys.argv[2]; DEV = sys.argv[3] if len(sys.argv) > 3 else "cuda"; os.makedirs(out, exist_ok=True)
I = json.load(open(art + "/index.json")); K = I["K"]; Hj = json.load(open(art + "/hadamard.json"))
fold = set(Hj["prism.hadamard.weight_names"]) | set(Hj["prism.hadamard.inverse_weight_names"]); sg = {}; o = 0
for w in Hj["prism.hadamard.sign_widths"]: sg[w] = torch.tensor(Hj["prism.hadamard.sign_values"][o:o + w], dtype=torch.float32, device=DEV); o += w
Hm = torch.tensor(scipy.linalg.hadamard(1024), dtype=torch.float32, device=DEV) / 32.0
q = np.frombuffer(lzma.decompress(open(art + "/scales.q.xz", "rb").read()), dtype=np.int8); rb = np.frombuffer(open(art + "/rowbase.bin", "rb").read(), dtype=np.float16)
tr = np.fromfile(art + "/trits.bin", dtype=np.uint8); qo = ro = to = 0; cur = {}; size = 0; sh = 0; wm = {}
NORMS = ("input_layernorm.weight", "post_attention_layernorm.weight", "language_model.norm.weight", "q_norm.weight", "k_norm.weight")
def hf(k): return k.replace("language_model.", "") if k.startswith("language_model.lm_head") else k.replace("language_model.model.", "model.language_model.")
def flush():
    global cur, size, sh
    if cur: sh += 1; fn = f"model-{sh:05d}.safetensors"; save_file(cur, out + "/" + fn, metadata={"format": "pt"}); wm.update({k: fn for k in cur}); cur = {}; size = 0
def put(k, t):
    global size
    cur[k] = t.contiguous(); size += t.numel() * t.element_size()
    if size > 4.5e9: flush()
for base, rows, cols, ng in I["index"]:
    n = rows * cols; nb = (n + 4) // 5
    p = torch.from_numpy(tr[to:to + nb].copy()).to(DEV).to(torch.int32); to += nb
    t = torch.stack([p % 3, (p // 3) % 3, (p // 9) % 3, (p // 27) % 3, (p // 81) % 3], 1).reshape(-1)[:n].reshape(rows, cols).float() - 1.0
    qq = torch.from_numpy(q[qo:qo + rows * ng].copy()).to(DEV).float().reshape(rows, ng); qo += rows * ng
    med = torch.from_numpy(rb[ro:ro + rows].copy()).to(DEV).float()[:, None]; ro += rows
    s = torch.exp2(med + qq / K)
    W = (t.reshape(rows, ng, 128) * s[..., None]).reshape(rows, cols)
    if base + ".weight" in fold: W = (W.reshape(rows, cols // 1024, 1024) @ Hm).reshape(rows, cols) * sg[cols]
    put(hf(base + ".weight"), W.to(torch.bfloat16).cpu())
for k, v in torch.load(art + "/small.pt").items():
    h = hf(k)
    if h.endswith("conv1d.weight") and v.ndim == 3 and v.shape[-1] == 1: v = v.permute(0, 2, 1)
    if h.endswith(NORMS): v = (v.float() - 1.0)
    put(h, v.float() if I["small_dtype"][k] == "torch.float32" or h.endswith(NORMS) else v.to(torch.bfloat16))
flush(); json.dump({"metadata": {}, "weight_map": wm}, open(out + "/model.safetensors.index.json", "w"))
