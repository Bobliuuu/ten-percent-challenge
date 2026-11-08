"""Shipped loader for the lossless Bonsai-8B-1bit pack. load(dir) -> HF Qwen3 model with the EXACT fp16 weights of the released checkpoint.
Format: bits.bin = per 128-group 16 bytes of sign bits (1 => +scale); scales.bin = fp16 group scale (max |w|); small.pt = all other tensors (fp16); index.json = [name, shape] of packed matrices."""
import json, numpy as np, torch
G = 128
def load(d, device="cuda"):
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
    missing = m.load_state_dict({k: v.to(device) for k, v in sd.items()}, strict=False)
    from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding
    m.model.rotary_emb = Qwen3RotaryEmbedding(config=cfg).to(device)
    if "lm_head.weight" not in sd: m.tie_weights()
    return m.eval(), missing
