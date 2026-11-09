#!/usr/bin/env python3
"""stats_embed: lossless fp16/bf16 exponent coding of embedding/head tables, tied/duplicate check (embed vs lm_head equality / cosine), unused-row fraction candidates.
python stats_embed.py MODEL_ID  -> results/stats_embed_<name>.json"""
import sys, json, os, torch
from huggingface_hub import get_safetensors_metadata, hf_hub_download
from safetensors import safe_open
m = sys.argv[1]; md = get_safetensors_metadata(m); where = {}
for fn, fm in md.files_metadata.items():
    for k in fm.tensors:
        if "embed_tokens" in k and "weight" in k and "visual" not in k: where["embed"] = (fn, k)
        if "lm_head" in k: where["head"] = (fn, k)
res = dict(model=m, found={k: v[1] for k, v in where.items()}); T = {}
for nm, (fn, k) in where.items():
    with safe_open(hf_hub_download(m, fn), "pt") as f: T[nm] = f.get_tensor(k)
def ent(x):
    raw = x.view(torch.int16).flatten().long() & 0xFFFF
    ex = (raw >> 7) & 255 if x.dtype == torch.bfloat16 else (raw >> 10) & 31
    nb = 7 if x.dtype == torch.bfloat16 else 10
    c = torch.bincount(ex).double(); p = c[c > 0] / c.sum(); h = float(-(p * p.log2()).sum())
    # also order-0 entropy of the whole 16-bit word and of sign+mantissa raw
    return dict(dtype=str(x.dtype), exp_H0=h, bits_exp_coded=1 + nb + h, saving_frac=1 - (1 + nb + h) / 16)
for nm, t in T.items():
    res[nm] = dict(shape=list(t.shape), **ent(t))
    nrm = t.float().norm(dim=1); res[nm]["row_norm_quantiles"] = [float(x) for x in torch.quantile(nrm[:200000], torch.tensor([.01, .1, .5, .9, .99]))]
    res[nm]["zero_rows"] = int((nrm == 0).sum())
if "embed" in T and "head" in T and T["embed"].shape == T["head"].shape:
    e, h = T["embed"].float(), T["head"].float(); res["embed_head_equal"] = bool(torch.equal(T["embed"], T["head"]))
    res["embed_head_rowcos_mean"] = float(torch.nn.functional.cosine_similarity(e[:50000], h[:50000], dim=1).mean())
os.makedirs("results", exist_ok=True); o = "results/stats_embed_" + m.replace("/", "_") + ".json"; json.dump(res, open(o, "w"), indent=1); print(json.dumps(res)[:800])
