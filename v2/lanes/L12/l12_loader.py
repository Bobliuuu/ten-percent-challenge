"""Reader for the l12_pack.py packed RTN4 body. unpack4 / dequant reproduce the engine's fake-quant tensors exactly (parity checked in run_pack.py)."""
import torch, glob, os, json
from safetensors.torch import load_file

def unpack4(b, n):
    b = b.to(torch.uint8); out = torch.stack([b & 15, b >> 4], 1).reshape(-1); return out[:n]

def dequant(d, name):
    shape = [int(x) for x in d[name + ".shape"]]; n = 1
    for s in shape: n *= s
    sc = d[name + ".sc"].float(); ng = sc.numel(); c = unpack4(d[name + ".q4"], n).float().reshape(*sc.shape[:-1], -1) if False else unpack4(d[name + ".q4"], n).float()
    g = n // ng; c = c.reshape(ng, g); z = unpack4(d[name + ".z4"], ng).float().reshape(ng, 1)
    return ((c - z) * sc.reshape(ng, 1)).reshape(shape)

def names(d): return sorted(k[:-6] for k in d if k.endswith(".shape"))

def load_dir(path):
    return {os.path.basename(f)[:-12]: load_file(f) for f in sorted(glob.glob(path + "/*.safetensors"))}
