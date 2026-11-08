"""Shipped decoder: pack dir -> Hugging Face model dir (bf16 safetensors). python decode_pack.py PACK_DIR OUT_DIR [device]
Reads qbody.json/qbody.bin/codebooks.bin (mixed-radix VQ indices), emb/head (tiered Lloyd-Max, fp16 or rANS-coded log scales), other.safetensors, config/tokenizer."""
import json, os, sys, shutil, zlib
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import packio as P
def main(pack, out, dev="cuda"):
    from safetensors.torch import save_file, load_file
    os.makedirs(out, exist_ok=True)
    for f in os.listdir(pack):
        if f.endswith((".json", ".txt", ".jinja")) and f not in ("qbody.json", "emb.json", "head.json", "model.safetensors.index.json"): shutil.copy(os.path.join(pack, f), out)
    hdr = json.load(open(f"{pack}/qbody.json")); seed = hdr["seed"]; SEEDP = (str(seed) + ":").encode() if seed else b""; D = hdr["D"]
    body = np.memmap(f"{pack}/qbody.bin", dtype=np.uint8, mode="r"); cbb = open(f"{pack}/codebooks.bin", "rb").read()
    cbs = {int(k): torch.from_numpy(np.frombuffer(cbb[o:o + l], dtype=np.float16).copy().reshape(int(k), D)).float().to(dev) for k, (o, l) in hdr["codebooks"].items()}
    import struct
    weights = {}; rins = {}; shard = {}; shard_n = 0; shards = []; wmap = {}; size = 0
    def flush():
        nonlocal shard, shard_n, size
        if not shard: return
        fn = f"model-{shard_n:05d}.safetensors"; save_file(shard, f"{out}/{fn}", metadata={"format": "pt"})
        for k in shard: wmap[k] = fn
        shard = {}; shard_n += 1; size = 0
    grp = {"self_attn.q_proj": "q_proj", "self_attn.k_proj": "q_proj", "self_attn.v_proj": "q_proj", "self_attn.o_proj": "o_proj", "mlp.gate_proj": "gate_proj", "mlp.up_proj": "gate_proj", "mlp.down_proj": "down_proj"}
    for n, K, sh, o, l, od, cols in hdr["tensors"]:
        li, nm = n.split(".", 1); gname = grp[nm]; sigma = struct.unpack(">f", bytes.fromhex(sh))[0]
        rin = P.Rot(cols, zlib.crc32(SEEDP + f"{li}.{gname}.in".encode()), dev); rout = P.Rot(od, zlib.crc32(SEEDP + f"{li}.{nm}.out".encode()), dev)
        idx = P.unpack_idx(bytes(body[o:o + l]), od * cols // D, K, dev).reshape(od, cols // D)
        W = P.decode_tensor(idx, cbs[K], sigma, rin, rout).to(torch.bfloat16).cpu().contiguous(); shard[f"model.layers.{n.split('.',1)[0]}.{nm}.weight"] = W; size += W.numel() * 2
        if size > 4e9: flush()
    for nm_, key in (("emb", "model.embed_tokens.weight"), ("head", "lm_head.weight")):
        if os.path.exists(f"{pack}/{nm_}.json"):
            shard[key] = P.dequant_embhead(P.load_embhead(f"{pack}/{nm_}"), dev); size += shard[key].numel() * 2
            if size > 4e9: flush()
    for k, v in load_file(f"{pack}/other.safetensors").items(): shard[k] = v
    flush()
    json.dump(dict(metadata={}, weight_map=wmap), open(f"{out}/model.safetensors.index.json", "w"))
    print("decoded ->", out, len(wmap), "tensors", flush=True)
if __name__ == "__main__": main(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else ("cuda" if torch.cuda.is_available() else "cpu"))
