#!/usr/bin/env python
"""`feedback` entry point of a TPQ1 artifact: closed-form in-place edit of existing output-embedding rows (v2/feedback_v3 fb5 `lmhead`, modes mean / lda), then
the touched rows are re-quantised on the tensor's own grid and re-encoded INTO model.tpq (same file, same probability tables, no new tensor, no new file).
   python feedback.py --artifact DIR --annotations ann.jsonl [--method lda|mean] [--alpha 0.2] [--ridge 0.1] [--stats-items f.jsonl]
ann rows = {id, query, tools, gold:[{name, arguments}]}.  Zero rows, or no site where the model is wrong => nothing is opened for writing.  Single forward passes only."""
import argparse, json, os, sys, time
# ---- loader (inlined) ----
import os, sys, io, json, zlib, struct, math, time, hashlib, mmap
import numpy as np
import torch

HB = 128
MAGIC = b"TPQ1\0\0\0\0"
BLOB = "model.tpq"
TARGET_SYM = 1 << 23            # ~8M symbols per rANS chunk (parallel decode unit)
_CT = None
TIMERS = {}


def _ct():
    global _CT
    if _CT is None:
        import constriction
        _CT = (constriction.stream.stack, constriction.stream.model)
    return _CT


# ------------------------------------------------------------------ rANS helpers
def _probs(counts):
    c = np.asarray(counts, dtype=np.float64); c = c + (c.sum() * 1e-9 + 1e-12)       # tiny floor: no zero-probability symbol
    return (c / c.sum()).astype(np.float32)


def ans_encode(sym, probs):
    stack, model = _ct(); c = stack.AnsCoder(); c.encode_reverse(np.ascontiguousarray(sym, dtype=np.int32), model.Categorical(np.asarray(probs, dtype=np.float32), perfect=False))
    return c.get_compressed().astype("<u4").tobytes()


def ans_decode(buf, n, probs):
    stack, model = _ct(); w = np.frombuffer(buf, dtype="<u4").astype(np.uint32)
    return stack.AnsCoder(w).decode(model.Categorical(np.asarray(probs, dtype=np.float32), perfect=False), n)


def pack_mant(mq, mbits):
    bits = ((mq[:, None].astype(np.uint16) >> np.arange(mbits, dtype=np.uint16)) & 1).astype(np.uint8).ravel()
    return np.packbits(bits).tobytes()


def unpack_mant(buf, n, mbits):
    bits = np.unpackbits(np.frombuffer(buf, dtype=np.uint8))[:n * mbits].reshape(n, mbits).astype(np.uint16)
    return (bits << np.arange(mbits, dtype=np.uint16)).sum(1).astype(np.uint16)


def encode_chunk(codes, scales, probs, eprobs, mbits):
    """codes uint8 [r,C]; scales fp16 [r,C/g] (positive, mantissa already rounded to mbits) -> (bytes, (lc,le,lm))."""
    b = np.ascontiguousarray(scales).astype(np.float16).view(np.uint16).ravel()
    assert not (b >> 15).any(), "negative scale"
    assert not (b & ((1 << (10 - mbits)) - 1)).any(), "scale mantissa has bits below mbits (writer is lossless-only)"
    e = (b >> 10) & 31; mq = (b & 1023) >> (10 - mbits)
    cb = ans_encode(codes.ravel(), probs); eb = ans_encode(e, eprobs); mb = pack_mant(mq, mbits)
    return cb + eb + mb, (len(cb), len(eb), len(mb))


def decode_chunk(buf, lens, nsym, nsc, probs, eprobs, mbits):
    lc, le, lm = lens
    codes = ans_decode(buf[:lc], nsym, probs).astype(np.uint8)
    e = ans_decode(buf[lc:lc + le], nsc, eprobs).astype(np.uint16); mq = unpack_mant(buf[lc + le:lc + le + lm], nsc, mbits)
    sc = ((e << 10) | (mq << (10 - mbits))).astype(np.uint16).view(np.float16)
    return codes, sc


def _dec_job(a):
    path, off, lens, nsym, nsc, probs, eprobs, mbits, pc, ps, oc, os_ = a
    with open(path, "rb") as f:
        f.seek(off); buf = f.read(sum(lens))
    codes, sc = decode_chunk(buf, lens, nsym, nsc, probs, eprobs, mbits)
    np.memmap(pc, dtype=np.uint8, mode="r+", offset=oc, shape=(nsym,))[:] = codes
    np.memmap(ps, dtype=np.float16, mode="r+", offset=os_, shape=(nsc,))[:] = sc
    return 0


def _scratch():
    import tempfile
    for d in ("/dev/shm", tempfile.gettempdir()):
        if os.path.isdir(d) and os.access(d, os.W_OK): return d


# ------------------------------------------------------------------ rotation + dequantisation (exactly the quantiser's `unrot`)
_HB = {}


def hada(dev):
    k = str(dev)
    if k not in _HB:
        Hm = torch.ones(1, 1)
        while Hm.shape[0] < HB: Hm = torch.cat([torch.cat([Hm, Hm], 1), torch.cat([Hm, -Hm], 1)], 0)
        _HB[k] = (Hm / math.sqrt(HB)).to(dev)
    return _HB[k]


def rot(M, sg):
    R, C = M.shape; return ((M * sg).view(R, C // HB, HB) @ hada(M.device)).reshape(R, C)


def unrot(M, sg):
    R, C = M.shape; return (M.view(R, C // HB, HB) @ hada(M.device)).reshape(R, C) * sg


def dequant(codes, scales, L, g, sg, dev=None, rows_per_call=4096):
    """codes uint8 [R,C], scales fp16 [R,C/g], sg float +-1 [C] or None -> bf16 [R,C] on dev. Bit-identical to the quantiser's simulated weight."""
    dev = dev or codes.device; R, C = codes.shape; half = (L - 1) / 2; out = torch.empty(R, C, dtype=torch.bfloat16, device=dev)
    sgn = None if sg is None else sg.to(dev).float()
    for r in range(0, R, rows_per_call):
        c = codes[r:r + rows_per_call].to(dev); s = scales[r:r + rows_per_call].to(dev)
        q = (c.float() - half) * s.float().repeat_interleave(g, 1)
        out[r:r + rows_per_call] = (q if sgn is None else unrot(q, sgn)).to(torch.bfloat16)
    return out


_DT = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32}
_DTN = {torch.bfloat16: "BF16", torch.float16: "F16", torch.float32: "F32"}


# ------------------------------------------------------------------ container
class Container:
    def __init__(self, path, workers=0):
        self.path = path; self.f = open(path, "rb"); self.f.seek(-16, 2); hl, mg = struct.unpack("<Q8s", self.f.read(16))
        assert mg == MAGIC, "bad footer"
        self.f.seek(-16 - hl, 2); self.h = json.loads(zlib.decompress(self.f.read(hl))); self.size = os.path.getsize(path)
        self.f.seek(0); assert self.f.read(8) == MAGIC, "bad magic"
        self.T = self.h["tensors"]; self.workers = workers; self._pool = None
        if workers > 1:
            import multiprocessing as mp
            try: self._pool = mp.get_context("fork").Pool(workers)
            except Exception: self._pool = None

    def close(self):
        if self._pool: self._pool.terminate(); self._pool = None
        self.f.close()

    def names(self): return list(self.T)

    def _chunk_pos(self, t):
        off = t["off"]; pos = []
        for i in range(0, len(t["lens"]), 3):
            ln = t["lens"][i:i + 3]; pos.append((off, ln)); off += sum(ln)
        return pos

    def chunk_rows(self, t, i):
        r0 = i * t["rpc"]; return r0, min(t["shape"][0], r0 + t["rpc"])

    def read_chunk(self, name, i):
        t = self.T[name]; off, ln = self._chunk_pos(t)[i]; self.f.seek(off); buf = self.f.read(sum(ln))
        r0, r1 = self.chunk_rows(t, i); R, C = t["shape"]; G = C // t["g"]
        return decode_chunk(buf, ln, (r1 - r0) * C, (r1 - r0) * G, t["probs"], t["eprobs"], t["mbits"])

    def read_q(self, name):
        """-> (codes uint8 torch [R,C], scales fp16 torch [R,C/g]) on CPU."""
        t = self.T[name]; R, C = t["shape"]; G = C // t["g"]; pos = self._chunk_pos(t); n = len(pos)
        if self._pool is None or n == 1:
            cs, ss = zip(*[self.read_chunk(name, i) for i in range(n)])
            return torch.from_numpy(np.concatenate(cs).reshape(R, C)), torch.from_numpy(np.concatenate(ss).reshape(R, G))
        import tempfile
        d = _scratch(); pc = os.path.join(d, f"tpq_c_{os.getpid()}_{id(t)}"); ps = os.path.join(d, f"tpq_s_{os.getpid()}_{id(t)}")
        try:
            np.memmap(pc, dtype=np.uint8, mode="w+", shape=(R * C,)).flush(); np.memmap(ps, dtype=np.float16, mode="w+", shape=(R * G,)).flush()
            jobs = []
            for i, (off, ln) in enumerate(pos):
                r0, r1 = self.chunk_rows(t, i)
                jobs.append((self.path, off, ln, (r1 - r0) * C, (r1 - r0) * G, t["probs"], t["eprobs"], t["mbits"], pc, ps, r0 * C, r0 * G * 2))
            self._pool.map(_dec_job, jobs, chunksize=1)
            codes = torch.from_numpy(np.fromfile(pc, dtype=np.uint8).reshape(R, C)); sc = torch.from_numpy(np.fromfile(ps, dtype=np.float16).reshape(R, G))
        finally:
            for p_ in (pc, ps):
                try: os.remove(p_)
                except OSError: pass
        return codes, sc

    def signs(self, key):
        s = self.h["signs"][key]; self.f.seek(s["off"]); b = np.frombuffer(self.f.read(s["len"]), dtype=np.uint8)
        return torch.from_numpy(np.unpackbits(b)[:s["n"]].astype(np.float32) * 2 - 1)

    def read_raw(self, name):
        t = self.T[name]; self.f.seek(t["off"]); b = bytearray(self.f.read(t["len"]))
        return torch.frombuffer(b, dtype=_DT[t["dtype"]]).reshape(t["shape"]).clone()

    def tensor(self, name, dev="cpu"):
        t = self.T[name]
        if t["kind"] == "alias": return self.tensor(t["of"], dev)
        if t["kind"] == "raw": return self.read_raw(name).to(dev)
        codes, sc = self.read_q(name)
        return dequant(codes.to(dev), sc.to(dev), t["L"], t["g"], None if t.get("rot") is None else self.signs(t["rot"]), dev)

    def manifest(self):
        return {n: [t["dtype"] if t["kind"] == "raw" else t["dtype"], t["shape"]] for n, t in self.T.items()}


def sha_tensor(x):
    return hashlib.sha256(x.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes() if x.dtype != torch.bfloat16 else x.detach().cpu().contiguous().view(torch.int16).numpy().tobytes()).hexdigest()


# ------------------------------------------------------------------ writer (also used by the in-place feedback edit)
class Writer:
    def __init__(self, path):
        self.path = path; self.f = open(path, "wb"); self.f.write(MAGIC); self.T = {}; self.S = {}

    def add_alias(self, name, of, dtype_name, shape):
        """tied checkpoints that store the same tensor twice (lm_head == embed_tokens): declared in the manifest, stored once"""
        self.T[name] = dict(kind="alias", of=of, dtype=dtype_name, shape=list(shape))

    def add_signs(self, key, sg):
        sg = (sg.cpu().numpy() > 0).astype(np.uint8); b = np.packbits(sg).tobytes(); self.S[key] = dict(off=self.f.tell(), len=len(b), n=int(sg.size)); self.f.write(b)

    def add_raw(self, name, x, dtype_name=None):
        x = x.detach().cpu().contiguous(); dn = dtype_name or _DTN[x.dtype]; b = x.view(torch.uint8).numpy().tobytes()
        self.T[name] = dict(kind="raw", dtype=dn, shape=list(x.shape), off=self.f.tell(), len=len(b)); self.f.write(b)

    def add_q(self, name, codes, scales, L, g, mbits, rot_key, dtype_name="BF16", tgt=None):
        tgt = tgt or TARGET_SYM
        codes = np.asarray(codes, dtype=np.uint8); scales = np.asarray(scales, dtype=np.float16); R, C = codes.shape; G = C // g
        assert scales.shape == (R, G) and codes.max() < L
        probs = _probs(np.bincount(codes.ravel(), minlength=L)); b = scales.view(np.uint16).ravel(); eprobs = _probs(np.bincount(((b >> 10) & 31).astype(np.int64), minlength=32))
        rpc = max(1, tgt // C); lens = []; off = self.f.tell(); stats = dict(codes=0, exp=0, mant=0)
        for r0 in range(0, R, rpc):
            r1 = min(R, r0 + rpc); blob, ln = encode_chunk(codes[r0:r1], scales[r0:r1], probs, eprobs, mbits); self.f.write(blob); lens += list(ln)
        self.T[name] = dict(kind="q", dtype=dtype_name, shape=[R, C], L=int(L), g=int(g), mbits=int(mbits), rot=rot_key, rpc=int(rpc), off=off, lens=lens,
                            probs=[float(x) for x in probs], eprobs=[float(x) for x in eprobs])
        return dict(codes=sum(lens[0::3]), exp=sum(lens[1::3]), mant=sum(lens[2::3]))

    def finish(self, meta=None):
        h = zlib.compress(json.dumps(dict(version=1, hb=HB, tensors=self.T, signs=self.S, meta=meta or {}), separators=(",", ":")).encode(), 9)
        self.f.write(h); self.f.write(struct.pack("<Q8s", len(h), MAGIC)); self.f.close(); return os.path.getsize(self.path)


def rewrite_with(path, replace, new_header_tensors=None):
    """Copy container -> path.tmp replacing chunks. replace = {(name, chunk_idx): (blob_bytes, (lc,le,lm))}. Untouched chunks are copied verbatim.
    Atomic os.replace at the end: same file name, zero added files."""
    c = Container(path); h = c.h; T = h["tensors"]; tmp = path + ".tmp"
    spans = []
    for n, t in T.items():
        if t["kind"] == "alias": continue
        if t["kind"] == "raw": spans.append((t["off"], t["len"], n, None))
        else:
            for i, (off, ln) in enumerate(c._chunk_pos(t)): spans.append((off, sum(ln), n, i))
    for k, s in h["signs"].items(): spans.append((s["off"], s["len"], "signs:" + k, None))
    spans.sort()
    newT = json.loads(json.dumps(T)); newS = json.loads(json.dumps(h["signs"])); out = open(tmp, "wb"); out.write(MAGIC); cur = 8
    newlens = {n: list(t["lens"]) for n, t in T.items() if t["kind"] == "q"}
    for off, ln, n, i in spans:
        c.f.seek(off); data = c.f.read(ln)
        if i is not None and (n, i) in replace:
            data, l3 = replace[(n, i)]; newlens[n][3 * i:3 * i + 3] = list(l3)
        if n.startswith("signs:"): newS[n[6:]]["off"] = cur
        elif i is None: newT[n]["off"] = cur
        elif i == 0: newT[n]["off"] = cur
        out.write(data); cur += len(data)
    for n, l in newlens.items(): newT[n]["lens"] = l
    hz = zlib.compress(json.dumps(dict(version=1, hb=HB, tensors=newT, signs=newS, meta=h.get("meta", {})), separators=(",", ":")).encode(), 9)
    out.write(hz); out.write(struct.pack("<Q8s", len(hz), MAGIC)); out.flush(); os.fsync(out.fileno()); out.close(); c.close(); os.replace(tmp, path)


# ------------------------------------------------------------------ RTN row re-quantisation (feedback re-encode) -- same scale search as the quantiser
PG = [1.0, .9, .8, .7, .62, .55, .5, .45, .4, .36, .32, .28, .25, .22, .19, .16, .13]


def round_scale_mbits(sc, mbits):
    """fp16 scale tensor -> mantissa rounded to mbits bits (round-half-up on the bit pattern), still fp16."""
    if mbits >= 10: return sc.half()
    b = sc.half().view(torch.int16).to(torch.int32) & 0xFFFF; sh = 10 - mbits
    b = (((b + (1 << (sh - 1))) >> sh) << sh) & 0xFFFF
    return torch.where(b >= 32768, b - 65536, b).to(torch.int16).view(torch.float16)


def qparams(x, Lv, mbits=10):
    half = (Lv - 1) / 2; am = x.abs().amax(1, keepdim=True).clamp(min=1e-8); best = None
    for p in PG:
        sc = round_scale_mbits(am * p / max(half, 0.5), mbits).float().clamp(min=2.0 ** -24)
        q = (torch.clamp(torch.round(x / sc + half), 0, Lv - 1) - half) * sc; e = ((q - x) ** 2).sum(1, keepdim=True)
        best = (e, sc) if best is None else (torch.where(e < best[0], e, best[0]), torch.where(e < best[0], sc, best[1]))
    return best[1]


def rtn_rows(W, Lv, g, mbits):
    """W float [r,C] (already rotated) -> (codes uint8 [r,C], scales fp16 [r,C/g])"""
    r, C = W.shape; x = W.reshape(-1, g); sc = qparams(x, Lv, mbits); half = (Lv - 1) / 2
    qi = torch.clamp(torch.round(x / sc + half), 0, Lv - 1)
    return qi.to(torch.uint8).reshape(r, C), sc.half().reshape(r, C // g)


def replace_rows(art_dir, tensor_name, new_rows, dev="cpu"):
    """new_rows {row_index: bf16/float row [C]} -> re-quantise those rows on the tensor's own grid (RTN, tensor's L/g/mbits, same rotation) and
    re-encode only the touched chunks with the SAME probability tables, inside the same file. Returns dict(chunks, rows, size_before, size_after)."""
    path = os.path.join(art_dir, BLOB); c = Container(path); t = c.T[tensor_name]; sg = None if t.get("rot") is None else c.signs(t["rot"]).to(dev)
    by = {}
    for r in new_rows: by.setdefault(r // t["rpc"], []).append(r)
    rep = {}
    for ci, rows in sorted(by.items()):
        codes, sc = c.read_chunk(tensor_name, ci); r0, _ = c.chunk_rows(t, ci)
        codes = codes.reshape(-1, t["shape"][1]).copy(); sc = sc.reshape(codes.shape[0], -1).copy()
        for r in rows:
            w = new_rows[r].to(dev).float().reshape(1, -1); wr = rot(w, sg) if sg is not None else w
            cq, sq = rtn_rows(wr, t["L"], t["g"], t["mbits"]); codes[r - r0] = cq[0].cpu().numpy(); sc[r - r0] = sq[0].cpu().numpy()
        rep[(tensor_name, ci)] = encode_chunk(codes, sc, t["probs"], t["eprobs"], t["mbits"])
    before = os.path.getsize(path); c.close(); rewrite_with(path, rep)
    return dict(chunks=len(rep), rows=len(new_rows), size_before=before, size_after=os.path.getsize(path))


# ------------------------------------------------------------------ model construction from the artifact
def load_model(art, dev="cuda", workers=None, log=print):
    from transformers import AutoConfig, AutoModelForCausalLM, GenerationConfig
    t0 = time.time(); torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    workers = workers if workers is not None else min(16, os.cpu_count() or 1)
    c = Container(os.path.join(art, BLOB), workers=workers)
    conf = AutoConfig.from_pretrained(art); conf._attn_implementation = "sdpa"
    with torch.device("meta"):
        try: m = AutoModelForCausalLM.from_config(conf, dtype=torch.bfloat16)
        except TypeError: m = AutoModelForCausalLM.from_config(conf, torch_dtype=torch.bfloat16)
    m = m.to(torch.bfloat16); m.to_empty(device=dev); params = dict(m.named_parameters()); seen = set(); tdec = 0.0
    for n in c.names():
        if n not in params:
            if c.T[n]["kind"] == "alias" and conf.tie_word_embeddings: seen.add(n); continue
            raise RuntimeError("tensor in artifact not in architecture: " + n)
        if c.T[n]["kind"] == "alias": continue
        t1 = time.time(); w = c.tensor(n, dev); assert list(w.shape) == list(params[n].shape), n
        params[n].data.copy_(w); seen.add(n); del w; tdec += time.time() - t1
    if conf.tie_word_embeddings: m.tie_weights(); seen |= {"lm_head.weight"}
    miss = set(params) - seen
    if miss: raise RuntimeError("parameters not covered by artifact: " + str(sorted(miss)[:5]))
    RC = type(m.model.rotary_emb)
    try: m.model.rotary_emb = RC(config=conf, device=dev)
    except TypeError: m.model.rotary_emb = RC(config=conf).to(dev)
    m.eval(); m.config.use_cache = True
    for p in m.parameters(): p.requires_grad_(False)
    try: m.generation_config = GenerationConfig.from_pretrained(art)
    except Exception: pass
    c.close(); TIMERS.update(load_total=time.time() - t0, decode_dequant=tdec, workers=workers); log(f"LOAD_DONE {time.time() - t0:.1f}s (decode+dequant {tdec:.1f}s) workers={workers}")
    return m, time.time() - t0


def export_bf16(art, out_dir, shard_bytes=5 << 30, dev="cuda"):
    """Dequantise the artifact into a plain bf16 safetensors shard set + index (for vLLM or any HF loader). Output dir must be OUTSIDE the artifact."""
    from safetensors.torch import save_file
    os.makedirs(out_dir, exist_ok=True); c = Container(os.path.join(art, BLOB), workers=min(16, os.cpu_count() or 1)); cur, cb, k, wm = {}, 0, 0, {}
    def flush():
        nonlocal cur, cb, k
        if cur:
            fn = f"model-{k:05d}.safetensors"; save_file(cur, os.path.join(out_dir, fn), metadata={"format": "pt"}); wm.update({n: fn for n in cur}); k += 1; cur, cb = {}, 0
    for n in c.names():
        w = c.tensor(n, dev).cpu(); cur[n] = w; cb += w.numel() * w.element_size()
        if cb >= shard_bytes: flush()
    flush(); json.dump(dict(metadata=dict(total_size=sum(os.path.getsize(os.path.join(out_dir, f)) for f in set(wm.values()))), weight_map=wm), open(os.path.join(out_dir, "model.safetensors.index.json"), "w"))
    for f in os.listdir(art):
        if f != BLOB and f.endswith((".json", ".txt", ".jinja")) and f != "tensor_manifest.json":
            import shutil; shutil.copy(os.path.join(art, f), os.path.join(out_dir, f))
    c.close()


# ------------------------------------------------------------------ prompt rendering + greedy generation (same code path as the harness' base_gen)
def render(tok, item):
    tools = [{"type": "function", "function": t} for t in item["tools"]]; msgs = [{"role": "user", "content": item["query"]}]
    try: return tok.apply_chat_template(msgs, tools=tools, add_generation_prompt=True, tokenize=False, enable_thinking=False)
    except TypeError: return tok.apply_chat_template(msgs, tools=tools, add_generation_prompt=True, tokenize=False)


def generate_with(model, tok, items, max_new_tokens=512, bs=8, prompts_out=None, dev="cuda", log=None):
    prompts = [render(tok, it) for it in items]
    if prompts_out: json.dump(prompts, open(prompts_out, "w"))
    out = [None] * len(items); order = sorted(range(len(items)), key=lambda i: len(prompts[i])); t0 = time.time(); ntok = 0
    with torch.no_grad():
        for s in range(0, len(order), bs):
            idx = order[s:s + bs]; enc = tok([prompts[i] for i in idx], return_tensors="pt", padding=True, add_special_tokens=False).to(dev)
            g = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tok.pad_token_id)
            for j, i in enumerate(idx):
                ids = g[j, enc.input_ids.shape[1]:]; ntok += int((ids != tok.pad_token_id).sum())
                out[i] = tok.decode(ids, skip_special_tokens=False).replace("<|endoftext|>", "").replace("<|im_end|>", "").strip()
            if log: log(f"gen {min(s + bs, len(order))}/{len(order)} {time.time() - t0:.0f}s")
    return out, dict(gen_seconds=time.time() - t0, new_tokens=ntok)

# ---- end loader ----


GENERIC = ["Hello, how are you today?", "Explain what a hash table is.", "Write a short poem about autumn.", "What is the capital of France?",
           "Summarise the plot of Romeo and Juliet in two sentences.", "How do I reverse a list in Python?", "Give me three tips for better sleep.",
           "Translate 'good morning' into Spanish.", "What causes rainbows?", "List the planets of the solar system."]


def edit_mean(H_by_tok, W, alpha, stats, ridge):
    out = {}
    for t, hs in H_by_tok.items():
        d = torch.stack(hs).mean(0) - stats["mu"]; d = d / d.norm(); w = W[t].float(); out[t] = (w + alpha * w.norm() * d).to(W.dtype)
    return out


def edit_lda(H_by_tok, W, alpha, stats, ridge):
    S = stats["sig"]; A = S + ridge * S.diagonal().mean() * torch.eye(S.shape[0], device=S.device); out = {}
    for t, hs in H_by_tok.items():
        d = torch.linalg.solve(A, torch.stack(hs).mean(0) - stats["mu"]); d = d / d.norm(); w = W[t].float(); out[t] = (w + alpha * w.norm() * d).to(W.dtype)
    return out


EDIT_FNS = {"mean": edit_mean, "lda": edit_lda}      # pluggable: fn(H_by_token, lm_head_weight, alpha, stats{mu,sig}, ridge) -> {row: new bf16 row}


def gold_text(g): return "\n".join("<tool_call>\n" + json.dumps(dict(name=x["name"], arguments=x["arguments"])) + "\n</tool_call>" for x in g)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--artifact", default=os.path.dirname(os.path.abspath(__file__))); ap.add_argument("--annotations", required=True)
    ap.add_argument("--method", default="lda"); ap.add_argument("--alpha", type=float, default=0.2); ap.add_argument("--ridge", type=float, default=0.1)
    ap.add_argument("--stats-items"); ap.add_argument("--device"); ap.add_argument("--workers", type=int)
    a = ap.parse_args(); anns = [json.loads(l) for l in open(a.annotations) if l.strip()]
    if not anns: print("FEEDBACK_NOOP no annotations"); return
    dev = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.artifact); model, _ = load_model(a.artifact, dev, a.workers, log=lambda *x: print(*x, file=sys.stderr, flush=True))
    head_name = "lm_head.weight" if "lm_head.weight" in model.state_dict() and not model.config.tie_word_embeddings else "model.embed_tokens.weight"
    W = model.lm_head.weight.data; sites = []
    with torch.no_grad():
        for it in anns:
            if not it.get("gold"): continue
            p = tok(render(tok, it), add_special_tokens=False).input_ids; g = tok(gold_text(it["gold"]) + "<|im_end|>", add_special_tokens=False).input_ids
            ids = torch.tensor([p + g], device=dev); h = model.model(ids).last_hidden_state[0]; lg = model.lm_head(h).float()
            for j, t in enumerate(g):
                pos = len(p) + j - 1
                if int(lg[pos].argmax()) != t: sites.append((t, h[pos].float().clone()))
        if not sites: print("FEEDBACK_NOOP model already right on every annotation token"); return
        hs = []
        texts = [tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True) for q in GENERIC]
        if a.stats_items: texts += [render(tok, json.loads(l)) for l in open(a.stats_items) if l.strip()][:120]
        for tx in texts:
            ids = tok(tx, return_tensors="pt", add_special_tokens=False).input_ids.to(dev); hs.append(model.model(ids).last_hidden_state[0].float())
        X = torch.cat(hs); stats = dict(mu=X.mean(0), sig=torch.cov(X.T))
        by = {}
        for t, h in sites: by.setdefault(t, []).append(h)
        edits = EDIT_FNS[a.method](by, W, a.alpha, stats, a.ridge)
    del model; torch.cuda.empty_cache() if dev == "cuda" else None
    r = replace_rows(a.artifact, head_name, {int(t): row.cpu() for t, row in edits.items()}, dev)
    print("FEEDBACK_DONE", json.dumps(dict(rows_edited=len(edits), sites=len(sites), tensor=head_name, **r)))


if __name__ == "__main__": main()
