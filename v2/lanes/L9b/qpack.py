"""L9b packed-artifact format: bit-packed grouped asymmetric RTN ints + fp16 scales + b-bit zero points, raw bf16 for small tensors, kept-expert pruning.
Files: manifest.json (tensor table + kept-expert lists + model class/config), body.bin (all tensor bytes), tokenizer files, config.json. Everything counts toward size."""
import json, os, math, numpy as np, torch

def pack_bits(v, b):
    """v: uint8 tensor (N,), values < 2**b, N % 8 == 0 -> uint8 bytes (N*b/8)"""
    sh = torch.arange(b, device=v.device, dtype=torch.uint8)
    bits = ((v.unsqueeze(-1) >> sh) & 1).reshape(-1, 8)
    w = (1 << torch.arange(8, device=v.device, dtype=torch.int32)).to(torch.int32)
    return (bits.to(torch.int32) * w).sum(-1).to(torch.uint8)

def unpack_bits(by, b, n):
    sh = torch.arange(8, device=by.device, dtype=torch.uint8)
    bits = ((by.unsqueeze(-1) >> sh) & 1).reshape(-1)[: n * b].reshape(n, b)
    w = (1 << torch.arange(b, device=by.device, dtype=torch.int32))
    return (bits.to(torch.int32) * w).sum(-1).to(torch.uint8)

def rtn_ints(w, bits, g):
    """w (..., C) float -> q uint8 (..., C), scale fp16 (..., C/g), zero uint8 (..., C/g) ; same math as moe_probe2.rtn"""
    sh = w.shape; wg = w.float().reshape(*sh[:-1], sh[-1] // g, g); mx = wg.amax(-1, keepdim=True); mn = wg.amin(-1, keepdim=True); qm = 2 ** bits - 1
    sc = ((mx - mn) / qm).clamp(min=1e-8).half(); scf = sc.float(); z = (-mn / scf).round().clamp(0, qm)
    q = (wg / scf + z).round().clamp(0, qm)
    return q.reshape(sh).to(torch.uint8), sc.squeeze(-1), z.squeeze(-1).to(torch.uint8)

def deq(q, sc, z, g, dtype=torch.bfloat16):
    sh = q.shape; qg = q.reshape(*sh[:-1], sh[-1] // g, g).float()
    return ((qg - z.float().unsqueeze(-1)) * sc.float().unsqueeze(-1)).reshape(sh).to(dtype)

class Writer:
    def __init__(s, d):
        os.makedirs(d, exist_ok=True); s.d = d; s.f = open(os.path.join(d, "body.bin"), "wb"); s.off = 0; s.T = {}
    def put(s, t):
        a = t.detach().contiguous().cpu().numpy().view(np.uint8).reshape(-1) if t.dtype != torch.uint8 else t.detach().cpu().numpy().reshape(-1)
        s.f.write(a.tobytes()); o = s.off; s.off += a.size; return [o, int(a.size)]
    def tensor(s, name, w, bits, g, chunk_rows=4096):
        """w: raw tensor (any shape); bits=0 -> raw bf16. quantised along the last dim, leading dims flattened to rows."""
        shape = list(w.shape)
        if not bits:
            s.T[name] = dict(shape=shape, kind="raw", raw=s.put(w.to(torch.bfloat16).view(torch.int16))); return
        w2 = w.reshape(-1, shape[-1]); R, C = w2.shape; assert C % g == 0 and (C * 1) % 8 == 0
        Q, S, Z = [], [], []
        qo = s.off; so = None
        e = dict(shape=shape, kind="q", bits=bits, g=g, R=R, C=C)
        # q ints, then scales, then zeros (three contiguous regions, each written in row chunks)
        qs, ss, zs = [], [], []
        for r0 in range(0, R, chunk_rows):
            q, sc, z = rtn_ints(w2[r0:r0 + chunk_rows].float(), bits, g)
            qs.append(pack_bits(q.reshape(-1), bits).cpu()); ss.append(sc.cpu()); zs.append(pack_bits(z.reshape(-1), bits if (z.numel() * bits) % 8 == 0 else bits).cpu() if False else z.reshape(-1).cpu())
        qb = torch.cat(qs); sb = torch.cat(ss).reshape(-1); zb = torch.cat(zs).reshape(-1)
        zpad = (-zb.numel()) % 8
        if zpad: zb = torch.cat([zb, torch.zeros(zpad, dtype=torch.uint8)])
        zbp = pack_bits(zb, bits)
        e["q"] = s.put(qb); e["s"] = s.put(sb.view(torch.int16)); e["z"] = s.put(zbp); e["nz"] = int(R * (C // g))
        s.T[name] = e
    def close(s, extra):
        s.f.close(); m = dict(tensors=s.T, **extra); json.dump(m, open(os.path.join(s.d, "manifest.json"), "w"), separators=(",", ":"))

def read_tensor(body, e, dev):
    """body: np.memmap uint8; returns bf16 tensor shape e['shape'] on dev"""
    sl = lambda ent: torch.from_numpy(np.array(body[ent[0]:ent[0] + ent[1]])).to(dev)
    if e["kind"] == "raw": return sl(e["raw"]).view(torch.int16).view(torch.bfloat16).reshape(e["shape"])
    b, g, R, C = e["bits"], e["g"], e["R"], e["C"]
    q = unpack_bits(sl(e["q"]), b, R * C).reshape(R, C); sc = sl(e["s"]).view(torch.int16).view(torch.float16).reshape(R, C // g)
    z = unpack_bits(sl(e["z"]), b, e["nz"]).reshape(R, C // g)
    return deq(q, sc, z, g).reshape(e["shape"])
