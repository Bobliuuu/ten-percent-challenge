"""Decode check: unpack a tensor from the packed body and compare with the dequantised checkpoint weight (bf16).
usage: python verify_pack.py PACK_DIR CKPT_DIR LAYER NAME [SEED]   e.g. ... /root/pk /root/q32_h 0 self_attn.k_proj"""
import sys, json, math, torch, numpy as np
from safetensors import safe_open
import glob, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import pack
pd, cd, L, name = sys.argv[1:5]; seed = int(sys.argv[5]) if len(sys.argv) > 5 else 0
J = json.load(open(pd + "/index.json"))["idx"][f"{L}.{name}"]; qi, sc = pack.unpack_tensor(pd, f"{L}.{name}")
Lv = J["L"]; half = (Lv - 1) / 2; R, C = qi.shape; g = J["g"]
Q = (torch.tensor(qi.astype(np.float32)) - half) * torch.tensor(sc.astype(np.float32)).repeat_interleave(g, 1)
key = {"self_attn.k_proj": "self_attn.q_proj", "self_attn.v_proj": "self_attn.q_proj", "mlp.up_proj": "mlp.gate_proj"}.get(name, name)
sg = (torch.randint(0, 2, (C,), generator=torch.Generator().manual_seed(1000 * int(L) + len(key) + 7919 * seed), dtype=torch.float32) * 2 - 1)
H = torch.ones(1, 1)
while H.shape[0] < 128: H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
H = H / math.sqrt(128); W = (Q.view(R, C // 128, 128) @ H).reshape(R, C) * sg
tn = f"model.layers.{L}.{name}.weight"; ref = None
for f in glob.glob(cd + "/*.safetensors"):
    with safe_open(f, "pt") as sf:
        if tn in sf.keys(): ref = sf.get_tensor(tn).float()
d = (W - ref).abs(); print(tn, "max abs diff", float(d.max()), "ref max", float(ref.abs().max()), "rel mean", float(d.mean() / ref.abs().mean()))
assert float(d.mean() / ref.abs().mean()) < 0.01, "decode mismatch"; print("DECODE_OK")
