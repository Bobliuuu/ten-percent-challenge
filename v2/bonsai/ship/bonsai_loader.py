"""Shipped loader for the lossless Bonsai-8B-1bit pack. load(dir) -> HF Qwen3 model with the EXACT fp16 weights of the released checkpoint.
Format: bits.bin = per 128-group 16 bytes of sign bits (1 => +scale); scales.bin = fp16 group scale (max |w|); small.pt = all other tensors (fp16); index.json = [name, shape] of packed matrices; optional patch.pt = fp16 replacement COLUMNS of the existing MLP down_proj matrices of the last layers (feedback; written by feedback.py)."""
import json, numpy as np, torch
G = 128
def load(d, device="cuda", patch="auto"):
    from transformers import AutoConfig, AutoModelForCausalLM
    cfg = AutoConfig.from_pretrained(d)
    bits = torch.from_numpy(np.fromfile(d + "/bits.bin", dtype=np.uint8)); sc = torch.from_numpy(np.fromfile(d + "/scales.bin", dtype=np.float16))
    small = torch.load(d + "/small.pt"); index = json.load(open(d + "/index.json")); sd = dict(small); ob = os_ = 0
    sh = torch.arange(8)
    for k, shape in index:
        ng = shape[0] * shape[1] // G; b = bits[ob:ob + ng * 16].view(ng, 16); s = sc[os_:os_ + ng]; ob += ng * 16; os_ += ng
        v = ((b[:, :, None] >> (7 - sh)) & 1).reshape(ng, G).half() * 2 - 1
        sd[k] = (v * s[:, None]).reshape(shape)
    with torch.device("meta"): m = AutoModelForCausalLM.from_config(cfg, torch_dtype=torch.float16, attn_implementation="sdpa")
    m = m.to_empty(device=device); 
    missing = m.load_state_dict(sd, strict=False)  # sd stays on CPU, copied tensor by tensor (fits 24GB GPUs)
    from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding
    m.model.rotary_emb = Qwen3RotaryEmbedding(config=cfg).to(device)
    if "lm_head.weight" not in sd: m.tie_weights()
    import os
    pf = d + "/patch.pt" if patch == "auto" else patch   # feedback patch: {"layers", "cols" (int16 column ids), "vals" (fp16 [hidden, r])}: columns of the EXISTING down_proj tensors
    if pf and os.path.exists(pf):
        P = torch.load(pf)
        for li, c, v in zip(P["layers"], P["cols"], P["vals"]): m.model.layers[li].mlp.down_proj.weight.data[:, c.long().to(device)] = v.to(device)
    return m.eval(), missing
