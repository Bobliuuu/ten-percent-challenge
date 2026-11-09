"""bank6q VQ artifact format (vq4_had_b2): REAL packer + loader for Hadamard-rotated, error-feedback (LDLQ/GPTQ) D4 K=256 Gaussian VQ experts.
Per expert tensor (gate_up_proj [k,1024,2048] / down_proj [k,2048,512]): fixed-rate uint8 index per 4 consecutive input-dim weights (2.000 bpw), one fp32 sigma per expert tensor,
ONE shared 256x4 fp32 codebook for the whole model (entry "__vq_cb__"), rotation = random +-1 signs (torch.Generator seed) then normalised Hadamard on the input axis (seed 1000+layer)
and on the output axis (seed 1000+layer+7919); signs are REGENERATED from the seeds stored in the manifest (no sign table shipped). Decode = cb[idx]*sigma -> inverse rotations -> bf16.
Everything else (attention/linear-attn/shared-expert 6-bit RTN, embedding+lm_head 4-bit RTN g128, router rows/norms/small tensors bf16) is exactly qpack. Self-contained: needs only torch/numpy/transformers + qpack.py."""
import json, os, math, sys, shutil
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qpack

_HC = {}
def hadamard(p, dev):
    k = (p, str(dev))
    if k not in _HC:
        Hm = torch.ones(1, 1, device=dev)
        while Hm.shape[0] < p: Hm = torch.cat([torch.cat([Hm, Hm], 1), torch.cat([Hm, -Hm], 1)], 0)
        _HC[k] = Hm / math.sqrt(p)
    return _HC[k]

class Rot:
    """identical to qlib.Rot(kind='had'): random signs, Hadamard on the power-of-2 part, (random orthogonal on the odd part; m==1 for every Qwen3.5-35B-A3B expert axis)"""
    def __init__(s, n, seed, dev):
        g = torch.Generator().manual_seed(seed); s.n = n
        s.sg = (torch.randint(0, 2, (n,), generator=g) * 2 - 1).float().to(dev)
        p = n & (-n); m = n // p; s.p, s.m = p, m
        assert m == 1, "odd part not supported in the shipped VQ format"
        s.Hd = hadamard(p, dev)
    def fwd(s, X):
        sh = X.shape; Y = X * s.sg; return torch.matmul(Y.reshape(-1, s.m, s.p), s.Hd).reshape(sh)
    def inv(s, X):
        sh = X.shape; Y = torch.matmul(X.reshape(-1, s.m, s.p), s.Hd); return Y.reshape(sh) * s.sg

def inv_rot(X, seed_in, seed_out):
    """X [E,R,C] float32 in the rotated domain -> original domain"""
    E, R, C = X.shape; ri = Rot(C, seed_in, X.device); ro = Rot(R, seed_out, X.device)
    return ri.inv(ro.inv(X.transpose(1, 2).contiguous()).transpose(1, 2).contiguous())

# ------------------------------------------------------------------ pack
def pack_vq_model(model, tok, kept, qdata, cb, outdir, attn_b=6, emb_b=4, head_b=4, g=128, code_files=(), extra=None):
    """kept: LongTensor [L,k]; qdata[l] = dict(gu=(idx uint8 [k,R,C/4], sig fp32 [k]), dn=(...)); cb: float32 [256,4]. Non-expert tensors are quantised here from the ORIGINAL bf16 weights."""
    import re
    W = qpack.Writer(outdir)
    cbt = cb.detach().float().contiguous().cpu(); W.T["__vq_cb__"] = dict(kind="rawf32", shape=list(cbt.shape), raw=W.put(cbt.view(torch.int32)))
    for n, p in model.named_parameters():
        d = p.data; m = re.match(r"model\.layers\.(\d+)\.mlp\.(experts\.(gate_up_proj|down_proj)|gate\.weight)$", n)
        if m:
            li = int(m.group(1)); ix = kept[li].to(d.device)
            if m.group(2) == "gate.weight": W.tensor(n, d.index_select(0, ix), 0, 0)
            else:
                key = "gu" if m.group(3) == "gate_up_proj" else "dn"; idx, sig = qdata[li][key]; k, R, C = len(ix), d.shape[1], d.shape[2]
                assert tuple(idx.shape) == (k, R, C // 4) and idx.dtype == torch.uint8 and sig.shape == (k,)
                W.T[n] = dict(kind="vq", shape=[k, R, C], D=4, K=256, idx=W.put(idx.contiguous()), sig=W.put(sig.float().contiguous().view(torch.int32)), seed_in=1000 + li, seed_out=1000 + li + 7919, rot="had")
        elif "embed_tokens" in n: W.tensor(n, d, emb_b, g)
        elif n.startswith("lm_head"): W.tensor(n, d, head_b, g)
        elif d.dim() == 2 and d.numel() >= 65536 and d.shape[-1] % g == 0: W.tensor(n, d, attn_b, g)
        else: W.tensor(n, d, 0, 0)
    cfg = model.config.to_dict(); cfg["num_experts"] = int(kept.shape[1])
    W.close(dict(model_cls=type(model).__name__, config_cls=type(model.config).__name__, config=cfg, kept=kept.tolist(), cfg=dict(format="vq4_had_b2", attn_b=attn_b, emb_b=emb_b, head_b=head_b, g=g), extra=extra or {}))
    tok.save_pretrained(outdir)
    for src in code_files: shutil.copy(src, os.path.join(outdir, os.path.basename(src)))

# ------------------------------------------------------------------ load
def _sl(body, ent, dev): return torch.from_numpy(np.array(body[ent[0]:ent[0] + ent[1]])).to(dev)

def read_codebook(body, m, dev):
    e = m["tensors"]["__vq_cb__"]; return _sl(body, e["raw"], dev).view(torch.int32).view(torch.float32).reshape(e["shape"])

def decode_vq(body, e, cb, dev, echunk=40):
    k, R, C = e["shape"]; idx = _sl(body, e["idx"], dev).reshape(k, R, C // 4); sig = _sl(body, e["sig"], dev).view(torch.int32).view(torch.float32)
    out = torch.empty(k, R, C, dtype=torch.bfloat16, device=dev)
    for a in range(0, k, echunk):
        X = cb[idx[a:a + echunk].long()].reshape(-1, R, C) * sig[a:a + echunk, None, None]
        out[a:a + echunk] = inv_rot(X, e["seed_in"], e["seed_out"]).to(torch.bfloat16)
    return out

def read_any(body, e, cb, dev):
    if e["kind"] == "vq": return decode_vq(body, e, cb, dev)
    return qpack.read_tensor(body, e, dev)

def load_packed_vq(adir, dev="cuda:0"):
    import transformers
    from transformers import AutoTokenizer
    m = json.load(open(os.path.join(adir, "manifest.json"))); body = np.memmap(os.path.join(adir, "body.bin"), dtype=np.uint8, mode="r")
    config = getattr(transformers, m["config_cls"]).from_dict(m["config"])
    old = torch.get_default_dtype(); torch.set_default_dtype(torch.bfloat16)
    with torch.device(dev): model = getattr(transformers, m["model_cls"])(config)
    torch.set_default_dtype(old)
    for p in model.parameters(): p.data = p.data.to(torch.bfloat16)   # parameters only: buffers (rotary inv_freq) must stay float32 as in the stock model
    model.eval()
    P = dict(model.named_parameters()); missing = [n for n in P if n not in m["tensors"]]; assert not missing, missing[:5]
    cb = read_codebook(body, m, dev)
    with torch.no_grad():
        for n, e in m["tensors"].items():
            if n == "__vq_cb__": continue
            P[n].data.copy_(read_any(body, e, cb, dev))
    for p in model.parameters(): p.requires_grad_(False)
    try:
        import fbcore; fbcore.load_carrier(model, adir)
    except ImportError: pass
    tok = AutoTokenizer.from_pretrained(adir)
    model.generation_config.eos_token_id = sorted({tok.convert_tokens_to_ids("<|im_end|>"), tok.eos_token_id} - {None}); tok.padding_side = "left"
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    return model, tok, m
