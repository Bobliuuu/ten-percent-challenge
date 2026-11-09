"""L5 packer: entropy-coded (rANS, per-tensor static histogram) scalar-quantised indices + coded group scales; measures ACTUAL bytes.
Used by v2/repair/dense_repair.py (cfg pack_dir). Format per tensor: [L counts as uint32][rANS words of indices][scale high bytes rANS][scale low bytes raw].
unpack_tensor() decodes back to (indices, scales) for round-trip verification."""
import numpy as np, json, os, time, constriction
def _enc(sym, L):
    h = np.bincount(sym, minlength=L).astype(np.int64); p = np.maximum(h, 1).astype(np.float64); p /= p.sum()
    c = constriction.stream.stack.AnsCoder(); c.encode_reverse(sym.astype(np.int32), constriction.stream.model.Categorical(p.astype(np.float32), perfect=False))
    return c.get_compressed().tobytes(), h
def _dec(buf, h, n):
    p = np.maximum(h, 1).astype(np.float64); p /= p.sum()
    c = constriction.stream.stack.AnsCoder(np.frombuffer(buf, dtype=np.uint32).copy()); return c.decode(constriction.stream.model.Categorical(p.astype(np.float32), perfect=False), n)
class Packer:
    def __init__(s, d): s.d = d; os.makedirs(d, exist_ok=True); s.f = open(d + "/body.bin", "wb"); s.idx = {}; s.bytes = dict(idx=0, scale=0, tables=0, other=0)
    def add(s, name, qi, sc, L, shape, g):
        """qi uint8 [R,C] numpy, sc fp16 [R, C//g] numpy"""
        t = time.time(); b1, h = _enc(qi.reshape(-1), L); u = sc.astype(np.float16).view(np.uint16).reshape(-1); hi = (u >> 8).astype(np.uint8); lo = (u & 255).astype(np.uint8)
        b2, h2 = _enc(hi, 256); tb = 4 * L + 4 * 256 + 32
        s.f.write(b1); s.f.write(b2); s.f.write(lo.tobytes())
        s.idx[name] = dict(L=L, shape=list(shape), g=g, n1=len(b1), n2=len(b2), n3=len(lo), h=h.tolist(), h2=h2.tolist())
        s.bytes["idx"] += len(b1); s.bytes["scale"] += len(b2) + len(lo); s.bytes["tables"] += tb
        return s.idx[name], len(b1), len(b2) + len(lo)
    def close(s, extra=None):
        s.f.close(); json.dump(dict(idx=s.idx, bytes=s.bytes, extra=extra or {}), open(s.d + "/index.json", "w"))
        return s.bytes
def unpack_tensor(d, name, meta=None, off=None):
    """decode tensor `name` -> (qi uint8 [R,C], scales fp16 [R,C//g]); reads body.bin sequentially by index order"""
    J = json.load(open(d + "/index.json"))["idx"]; o = 0
    for k, m in J.items():
        n = m["n1"] + m["n2"] + m["n3"]
        if k == name: break
        o += n
    m = J[name]; raw = open(d + "/body.bin", "rb").read()[o:o + m["n1"] + m["n2"] + m["n3"]]
    R, C = m["shape"]; qi = np.asarray(_dec(raw[:m["n1"]], np.array(m["h"]), R * C), dtype=np.uint8).reshape(R, C)
    hi = np.asarray(_dec(raw[m["n1"]:m["n1"] + m["n2"]], np.array(m["h2"]), R * C // m["g"]), dtype=np.uint8); lo = np.frombuffer(raw[m["n1"] + m["n2"]:], dtype=np.uint8)
    sc = ((hi.astype(np.uint16) << 8) | lo).view(np.float16).reshape(R, C // m["g"]); return qi, sc
if __name__ == "__main__":   # self test
    rng = np.random.RandomState(0); R, C, g, L = 64, 1024, 256, 3
    qi = rng.choice(L, size=(R, C), p=[.3, .4, .3]).astype(np.uint8); sc = (rng.rand(R, C // g) * .1 + .01).astype(np.float16)
    P = Packer("/tmp/ptest"); P.add("t", qi, sc, L, (R, C), g); b = P.close(); q2, s2 = unpack_tensor("/tmp/ptest", "t")
    assert (q2 == qi).all() and (s2 == sc).all(); print("roundtrip ok", b, "bits/idx", 8 * b["idx"] / qi.size)
