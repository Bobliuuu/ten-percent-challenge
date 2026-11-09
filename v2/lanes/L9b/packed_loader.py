"""Standalone loader for the L9b packed artifact: builds the pruned Qwen3.5 MoE from manifest.json (config + kept-expert count), skips weight init and the HF checkpoint entirely,
and decodes body.bin tensor by tensor on the GPU (unpack bits -> dequantise -> bf16, no dequantised checkpoint is ever written). Peak GPU memory = the bf16 text model (~31 GB) + one tensor.
Usage: model, tok, manifest = packed_loader.load(ARTIFACT_DIR, dev="cuda:0")   (then fbcore.load_carrier is applied if fb.json exists)"""
import json, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qpack

def _read(f, ent, dev):
    off, n = ent; f.seek(off); return torch.from_numpy(np.frombuffer(f.read(n), dtype=np.uint8).copy()).to(dev)

def read_tensor(f, e, dev, chunk_rows=8192):
    if e["kind"] == "raw": return _read(f, e["raw"], dev).view(torch.int16).view(torch.bfloat16).reshape(e["shape"])
    b, g, R, C = e["bits"], e["g"], e["R"], e["C"]
    sc = _read(f, e["s"], dev).view(torch.int16).view(torch.float16).reshape(R, C // g); z = qpack.unpack_bits(_read(f, e["z"], dev), b, e["nz"]).reshape(R, C // g)
    out = torch.empty(R, C, dtype=torch.bfloat16, device=dev); rb = C * b // 8   # bytes per row; rows are byte-aligned, so unpack row chunks (keeps the transient small)
    for r0 in range(0, R, chunk_rows):
        r1 = min(R, r0 + chunk_rows); qb = _read(f, [e["q"][0] + r0 * rb, (r1 - r0) * rb], dev)
        out[r0:r1] = qpack.deq(qpack.unpack_bits(qb, b, (r1 - r0) * C).reshape(r1 - r0, C), sc[r0:r1], z[r0:r1], g)
    return out.reshape(e["shape"])

def load(adir, dev="cuda:0", init=False):
    import transformers
    from transformers import AutoTokenizer
    m = json.load(open(os.path.join(adir, "manifest.json")))
    config = getattr(transformers, m["config_cls"]).from_dict(m["config"])
    config.dtype = torch.bfloat16
    for sub in ("text_config",):
        if hasattr(config, sub) and getattr(config, sub) is not None: getattr(config, sub).dtype = torch.bfloat16
    old = torch.get_default_dtype(); torch.set_default_dtype(torch.bfloat16)
    try:
        if init:
            with torch.device(dev): model = getattr(transformers, m["model_cls"])(config)
        else:
            from transformers.initialization import no_init_weights
            with torch.device(dev), no_init_weights(): model = getattr(transformers, m["model_cls"])(config)
    finally: torch.set_default_dtype(old)
    if os.environ.get("PL_DEBUG"): print("after construct GB", torch.cuda.memory_allocated() / 2**30, "peak", torch.cuda.max_memory_allocated() / 2**30, next(model.parameters()).dtype, flush=True)
    model = model.to(torch.bfloat16).eval()   # parameters are already bf16 (no copy); this also casts float buffers (rotary inv_freq) to bf16 exactly like core.load_packed, which is the path all sealed numbers were scored with; P = dict(model.named_parameters())
    missing = [n for n in P if n not in m["tensors"]]; assert not missing, missing[:5]
    with torch.no_grad(), open(os.path.join(adir, "body.bin"), "rb") as f:
        for n, e in m["tensors"].items():
            if n in P: P[n].data.copy_(read_tensor(f, e, dev))
    if os.environ.get("PL_DEBUG"): print("after load GB", torch.cuda.memory_allocated() / 2**30, "peak", torch.cuda.max_memory_allocated() / 2**30, flush=True)
    for p in model.parameters(): p.requires_grad_(False)
    try:
        import fbcore; fbcore.load_carrier(model, adir)
    except ImportError: pass
    tok = AutoTokenizer.from_pretrained(adir); tok.padding_side = "left"
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    return model, tok, m
