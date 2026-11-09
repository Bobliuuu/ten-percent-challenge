"""L10 pack format (real entropy-coded body) + standalone reader/dequantiser.
File = [payload sections ...][zlib(JSON header)][u64 header_len][b"L10PACK1"].
Entry kinds:
  q    : scalar-quantised matrix. symbols (uint8 levels) range-coded with ONE per-tensor categorical (counts stored in the header);
         fp16 group scales: high byte range-coded (per-tensor categorical), low byte stored raw. Dequantisation (exactly the GPTQ simulation):
         Q = (level - (L-1)/2) * scale ; W = bf16( unrot(Q) ),  unrot = block-Hadamard(128) then column sign vector (seeded RNG).
  raw  : tensor stored as-is (bf16 or fp32 bytes)
  file : tokenizer / config bytes
Needs: torch, numpy, constriction (pip install constriction)."""
import json, struct, zlib, hashlib, math, os
import numpy as np
MAGIC = b"L10PACK1"; HB = 128

def _probs(counts):
    c = counts.astype(np.float64) + 1.0
    return (c / c.sum()).astype(np.float32)

def rc_encode(sym, K):
    import constriction
    counts = np.bincount(sym, minlength=K).astype(np.uint32)
    enc = constriction.stream.queue.RangeEncoder()
    enc.encode(np.ascontiguousarray(sym, dtype=np.int32), constriction.stream.model.Categorical(_probs(counts), perfect=False))
    return enc.get_compressed().tobytes(), counts

def rc_decode(buf, counts, n):
    import constriction
    dec = constriction.stream.queue.RangeDecoder(np.frombuffer(buf, dtype=np.uint32))
    return dec.decode(constriction.stream.model.Categorical(_probs(np.asarray(counts, dtype=np.uint32)), perfect=False), n)

class Writer:
    def __init__(s, path): s.f = open(path, "wb"); s.pos = 0; s.ents = []; s.stats = dict(sym=0, scale=0, raw=0, file=0)
    def _w(s, b): off = s.pos; s.f.write(b); s.pos += len(b); return off, len(b)
    def add_q(s, name, qi, sc, Lv, g, seed, shape):
        """qi: uint8 [R*C] levels (row-major), sc: fp16 [R*C/g] group scales (row-major [R, C/g])"""
        qi = np.asarray(qi, dtype=np.uint8).reshape(-1); sbytes, cnt = rc_encode(qi.astype(np.int32), Lv)
        u = np.asarray(sc, dtype=np.float16).reshape(-1).view(np.uint16); hi = (u >> 8).astype(np.int32); lo = (u & 255).astype(np.uint8)
        hb, hcnt = rc_encode(hi, 256)
        o1 = s._w(sbytes); o2 = s._w(hb); o3 = s._w(lo.tobytes())
        s.ents.append(dict(name=name, kind="q", shape=list(shape), Lv=Lv, g=g, seed=seed, counts=cnt.tolist(), hcounts=hcnt.tolist(), nsc=int(u.size), sym=o1, hi=o2, lo=o3))
        s.stats["sym"] += o1[1]; s.stats["scale"] += o2[1] + o3[1]
    def add_raw(s, name, tensor):
        import torch
        t = tensor.detach().cpu().contiguous(); dt = {torch.bfloat16: "bf16", torch.float32: "fp32", torch.float16: "fp16"}[t.dtype]
        b = (t.view(torch.int16) if t.dtype != torch.float32 else t).numpy().tobytes(); o = s._w(b)
        s.ents.append(dict(name=name, kind="raw", dtype=dt, shape=list(t.shape), data=o)); s.stats["raw"] += o[1]
    def add_file(s, name, b): o = s._w(b); s.ents.append(dict(name=name, kind="file", data=o)); s.stats["file"] += o[1]
    def close(s, meta=None):
        h = zlib.compress(json.dumps(dict(entries=s.ents, meta=meta or {})).encode(), 9); s.f.write(h); s.f.write(struct.pack("<Q", len(h))); s.f.write(MAGIC); s.f.close()
        s.stats["header"] = len(h) + 16; return s.stats

def read_header(path):
    with open(path, "rb") as f:
        f.seek(-16, 2); n, = struct.unpack("<Q", f.read(8)); assert f.read(8) == MAGIC
        f.seek(-16 - n, 2); return json.loads(zlib.decompress(f.read(n)))

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""): h.update(b)
    return h.hexdigest()

def tensor_sha(t):
    import torch
    t = t.detach().cpu().contiguous(); return hashlib.sha256((t.view(torch.int16) if t.dtype != torch.float32 else t).numpy().tobytes()).hexdigest()

def _hb(dev):
    import torch
    Hm = torch.ones(1, 1)
    while Hm.shape[0] < HB: Hm = torch.cat([torch.cat([Hm, Hm], 1), torch.cat([Hm, -Hm], 1)], 0)
    return (Hm / math.sqrt(HB)).to(dev)

def signs(seed, C, dev):
    import torch
    return (torch.randint(0, 2, (C,), generator=torch.Generator().manual_seed(seed), dtype=torch.float32) * 2 - 1).to(dev)

def unrot(M, sg, Hb):
    R, C = M.shape; return ((M.view(R, C // HB, HB) @ Hb).reshape(R, C)) * sg

def dequant(e, f, dev="cuda", Hb=None):
    """-> bf16 tensor on dev, bit-identical to the packer's in-process weight (same GPU class)"""
    import torch
    R, C = e["shape"]; Lv = e["Lv"]; g = e["g"]; Hb = Hb if Hb is not None else _hb(dev)
    f.seek(e["sym"][0]); sb = f.read(e["sym"][1]); qi = rc_decode(sb, e["counts"], R * C).astype(np.uint8)
    f.seek(e["hi"][0]); hb = f.read(e["hi"][1]); hi = rc_decode(hb, e["hcounts"], e["nsc"]).astype(np.uint16)
    f.seek(e["lo"][0]); lo = np.frombuffer(f.read(e["lo"][1]), dtype=np.uint8).astype(np.uint16)
    sc = ((hi << 8) | lo).astype(np.uint16).view(np.float16)
    qi = torch.from_numpy(qi).to(dev).float().view(R, C // g, g); sc = torch.from_numpy(sc.copy()).to(dev).float().view(R, C // g, 1)
    Q = ((qi - (Lv - 1) / 2) * sc).view(R, C)
    return unrot(Q, signs(e["seed"], C, dev), Hb).to(torch.bfloat16)

def load_state(path, dev="cuda", only=None):
    """decode the whole pack -> (state_dict of bf16/fp32 tensors on dev, files {name: bytes}, meta)"""
    import torch
    H = read_header(path); sd = {}; files = {}; Hb = _hb(dev)
    with open(path, "rb") as f:
        for e in H["entries"]:
            if only and e["name"] not in only: continue
            if e["kind"] == "q": sd[e["name"]] = dequant(e, f, dev, Hb)
            elif e["kind"] == "raw":
                f.seek(e["data"][0]); b = f.read(e["data"][1]); dt = {"bf16": np.int16, "fp16": np.int16, "fp32": np.float32}[e["dtype"]]
                t = torch.from_numpy(np.frombuffer(b, dtype=dt).copy()).view(e["shape"])
                sd[e["name"]] = (t.view(torch.bfloat16) if e["dtype"] == "bf16" else t.view(torch.float16) if e["dtype"] == "fp16" else t).to(dev)
            else: f.seek(e["data"][0]); files[e["name"]] = f.read(e["data"][1])
    return sd, files, H["meta"]

def build_model(path, dev="cuda", workdir="/tmp/l10body"):
    """pack -> (model on dev in bf16, tokenizer). Writes config/tokenizer files to workdir (needed by transformers)."""
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    sd, files, meta = load_state(path, dev); os.makedirs(workdir, exist_ok=True)
    for n, b in files.items(): open(os.path.join(workdir, n), "wb").write(b)
    cfg = AutoConfig.from_pretrained(workdir)
    with torch.device(dev): m = AutoModelForCausalLM.from_config(cfg, torch_dtype=torch.bfloat16)
    r = m.load_state_dict(sd, strict=False); assert not r.missing_keys and not r.unexpected_keys, (r.missing_keys[:5], r.unexpected_keys[:5])
    del sd; sd = None
    m = m.to(dev).eval(); tok = AutoTokenizer.from_pretrained(workdir); return m, tok, meta

def write_hf_dir(path, out):
    """pack -> HF directory (safetensors + config + tokenizer) for tools that take a model path"""
    from safetensors.torch import save_file
    sd, files, meta = load_state(path, "cuda"); os.makedirs(out, exist_ok=True)
    for n, b in files.items(): open(os.path.join(out, n), "wb").write(b)
    save_file({k: v.cpu().contiguous() for k, v in sd.items()}, os.path.join(out, "model.safetensors"), metadata={"format": "pt"}); return meta

def replace_raw(pack_in, pack_out, tensors):
    """exact-size in-place replacement of raw entries (feedback edit of fp16-kept tensors): copies pack_in -> pack_out and overwrites the byte ranges"""
    import shutil, torch
    shutil.copyfile(pack_in, pack_out); H = read_header(pack_in); by = {e["name"]: e for e in H["entries"]}
    with open(pack_out, "r+b") as f:
        for n, t in tensors.items():
            e = by[n]; assert e["kind"] == "raw" and e["dtype"] == "bf16"; t = t.detach().cpu().to(torch.bfloat16).contiguous(); assert list(t.shape) == e["shape"]
            f.seek(e["data"][0]); f.write(t.view(torch.int16).numpy().tobytes())
