"""Packed artifact format for the L3 MoE entry: one weights.bin (b-bit packed codes, fp16 scales, packed zero-points, fp16 plain tensors) + index.json.
Everything shipped (weights.bin, index.json, config.json, tokenizer files, answer.py/feedback.py/runtime) is counted by MANIFEST.json."""
import numpy as np, json, os, torch

def pack_bits(q, bits):
    a = np.asarray(q, dtype=np.uint8).reshape(-1); b = ((a[:, None] >> np.arange(bits, dtype=np.uint8)) & 1).astype(np.uint8).reshape(-1)
    return np.packbits(b, bitorder="little").tobytes()

def unpack_bits(buf, bits, n):
    b = np.unpackbits(np.frombuffer(buf, dtype=np.uint8), bitorder="little")[: n * bits].reshape(n, bits)
    return (b << np.arange(bits, dtype=np.uint8)).sum(1).astype(np.uint8)

class Writer:
    def __init__(s, path): s.f = open(path, "wb"); s.idx = {}; s.pos = 0
    def _w(s, b): off = s.pos; s.f.write(b); s.pos += len(b); return off, len(b)
    def add_q(s, name, q, sc, z, bits, g):
        q = q.cpu().numpy(); sc = sc.cpu().numpy().astype(np.float16); z = z.cpu().numpy()
        s.idx[name] = dict(kind="q", shape=list(q.shape), bits=bits, g=g, q=s._w(pack_bits(q, bits)), s=s._w(sc.tobytes()), z=s._w(pack_bits(z, bits)))
    def add_fp16(s, name, t): t = t.cpu().to(torch.float16).numpy(); s.idx[name] = dict(kind="f16", shape=list(t.shape), d=s._w(t.tobytes()))
    def close(s, path): s.f.close(); json.dump(s.idx, open(path, "w"))

def read_tensor(mm, e):
    """returns float32 torch tensor (dequantised for kind q; NOT rotated back)"""
    if e["kind"] == "f16": o, n = e["d"]; return torch.from_numpy(np.frombuffer(mm[o:o + n], dtype=np.float16).reshape(e["shape"]).copy()).float()
    sh = e["shape"]; bits = e["bits"]; g = e["g"]; N = int(np.prod(sh))
    q = unpack_bits(mm[e["q"][0]:e["q"][0] + e["q"][1]], bits, N).reshape(sh)
    sc = np.frombuffer(mm[e["s"][0]:e["s"][0] + e["s"][1]], dtype=np.float16).reshape(*sh[:-1], sh[-1] // g)
    ng = int(np.prod(sc.shape)); z = unpack_bits(mm[e["z"][0]:e["z"][0] + e["z"][1]], bits, ng).reshape(sc.shape)
    qq = torch.from_numpy(q).float().reshape(*sh[:-1], sh[-1] // g, g); return ((qq - torch.from_numpy(z).float()[..., None]) * torch.from_numpy(sc).float()[..., None]).reshape(sh)
