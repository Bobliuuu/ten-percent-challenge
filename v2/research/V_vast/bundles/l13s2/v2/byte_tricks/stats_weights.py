#!/usr/bin/env python3
"""stats_weights: byte accounting on REAL tensors of the lanes' start models (no generation). For each model:
 (A) structure from HF metadata only (get_safetensors_metadata): bytes by tensor family (embed, head, linear, expert, norm/bias, visual/audio/mtp), tied/duplicate shapes
 (B) tokenizer/config/code file sizes and what xz/zstd saves
 (C) on sampled linear tensors from 1-2 downloaded shards: Hadamard + uniform RTN (L levels) index entropy and the scale-table cost under fp16 / raw8 / log-delta / row+col coding for g=128/256/512 and shared-row variants
 (D) lossless exponent coding of fp16 remainder tensors (norms, biases): entropy(exponent)+mantissa bits vs 16
 (E) already-discrete models (Bonsai/ternary): scale table (fp16 values) entropy, distinct values, bias==-scale check
python stats_weights.py MODEL_ID [--shards N] [--tensors K] [--discrete] [--out results/stats_<name>.json]"""
import argparse, json, os, sys, glob, lzma, math, struct, time
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from btcore import *

def fam(name):
    n = name.lower()
    if "embed_tokens" in n or n.endswith("wte.weight"): return "embed"
    if "lm_head" in n: return "head"
    if any(k in n for k in ("visual", "vision", "audio", "image", "mm_", "multi_modal", "projector")): return "modality"
    if "mtp" in n: return "mtp"
    if "experts" in n and "gate" not in n.split("experts")[-1][:6] or ".experts." in n: return "expert"
    if "norm" in n or n.endswith(".bias") or "layernorm" in n: return "norm_bias"
    if n.endswith(".weight") and "layers" in n: return "linear"
    return "other"

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("model"); ap.add_argument("--shards", type=int, default=1); ap.add_argument("--tensors", type=int, default=24)
    ap.add_argument("--discrete", action="store_true"); ap.add_argument("--skip", type=int, default=0); ap.add_argument("--out"); ap.add_argument("--dev", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args(); from huggingface_hub import HfApi, get_safetensors_metadata, hf_hub_download, snapshot_download
    res = dict(model=a.model); api = HfApi(); info = api.model_info(a.model, files_metadata=True)
    res["files"] = {s.rfilename: s.size for s in info.siblings}
    # (A) structure
    try:
        md = get_safetensors_metadata(a.model); fb = collections.defaultdict(lambda: [0, 0]) if False else {}
        tot = 0; shapes = {}
        for fn, m in md.files_metadata.items():
            for k, t in m.tensors.items():
                n = 1
                for d in t.shape: n *= d
                f = fam(k); fb.setdefault(f, [0, 0]); fb[f][0] += n; fb[f][1] += 1; tot += n; shapes.setdefault((tuple(t.shape), t.dtype), []).append(k)
        res["params_total"] = tot; res["family_params"] = {k: v[0] for k, v in fb.items()}; res["family_pct_of_base"] = {k: 100 * v[0] / tot for k, v in fb.items()}
        res["dup_shape_candidates"] = {f"{k[0]}": len(v) for k, v in shapes.items() if len(v) > 1 and ("embed" in v[0] or "lm_head" in v[0])}
        res["has_lm_head"] = any("lm_head" in k for _, m in md.files_metadata.items() for k in m.tensors)
    except Exception as e: res["meta_err"] = repr(e)
    # (B) tokenizer/config/code files
    tdir = snapshot_download(a.model, allow_patterns=["tokenizer*", "vocab*", "merges*", "config.json", "generation_config.json", "*.jinja", "special_tokens_map.json", "added_tokens.json", "preprocessor_config.json", "*.index.json", "*.py"])
    cat = {}
    for f in os.listdir(tdir):
        p = os.path.join(tdir, f)
        if os.path.isfile(p):
            b = open(p, "rb").read(); cat[f] = dict(bytes=len(b), xz=len(lzma.compress(b, preset=9 | lzma.PRESET_EXTREME)))
            try:
                import zstandard; cat[f]["zstd19"] = len(zstandard.ZstdCompressor(level=19).compress(b))
            except Exception: pass
    res["small_files"] = cat; res["small_files_total"] = sum(v["bytes"] for v in cat.values()); res["small_files_xz_total"] = sum(v["xz"] for v in cat.values())
    # (C/D/E) tensors from shards
    idx = [f for f in res["files"] if f.endswith(".safetensors")]; idx.sort(); pick = idx[a.skip: a.skip + a.shards] if a.shards else []
    from safetensors import safe_open
    lin = []; rem = []; out = dict(linear=[], remainder=[], discrete=[])
    for sh in pick:
        p = hf_hub_download(a.model, sh)
        with safe_open(p, "pt") as f:
            keys = list(f.keys())
            for k in keys:
                t = f.get_slice(k); shp = t.get_shape()
                if len(shp) == 2 and min(shp) >= 256 and fam(k) in ("linear", "expert") and len([1 for _ in lin]) < a.tensors * 4: lin.append((sh, k))
                elif len(shp) == 1 and len(rem) < 400: rem.append((sh, k))
        shard_path = p
    # sample
    import random; random.Random(0).shuffle(lin); lin = lin[: a.tensors]; dev = a.dev; P = {}
    for sh, k in lin + rem: P.setdefault(sh, []).append(k)
    cache = {}
    for sh, ks in P.items():
        with safe_open(hf_hub_download(a.model, sh), "pt") as f:
            for k in ks: cache[k] = f.get_tensor(k)
    for sh, k in lin:
        W = cache[k].to(dev).float(); R, C = W.shape
        if a.discrete:
            # group scales of an already-low-bit tensor: max|w| per 128 contiguous input cols; values are scale*{-1,0,1}(+bias)
            g = 128; Wg = W[:, :C // g * g].reshape(R, -1, g); s = Wg.abs().amax(-1); nz = (Wg != 0).float().mean().item()
            vals = Wg / s.clamp(min=1e-12)[..., None]; lev = torch.unique(vals.round(decimals=3)).numel()
            sf16 = s.half(); raw = sf16.view(torch.int16).flatten().long() & 0xFFFF; ex = (raw >> 10) & 31
            sb = dict(fp16=16.0 * s.numel())
            codes = torch.round(torch.log2(s.clamp(min=1e-12)) * 16).to(torch.int32); sb.update(scale_bits_all(codes, s))
            out["discrete"].append(dict(t=k, R=R, C=C, nz_frac=nz, n_levels=int(lev), scale_distinct_fp16=int(torch.unique(raw).numel()), scale_fp16_H0=H0_bits(raw) / s.numel(), scale_exp_H0=H0_bits(ex) / s.numel(),
                                        scale_bits_per_group={k2: v / s.numel() for k2, v in sb.items()}, n=W.numel()))
            continue
        row = dict(t=k, R=R, C=C, n=W.numel())
        Q = blockQ(C, 11, dev); Wr = W @ Q.T
        for L in (3, 5, 8):
            for g in (128, 256, 512):
                if C % g: continue
                for share in (1, 4):
                    if R % share: continue
                    q1 = Q1(L, g, share, "log16"); rtn(Wr, q1); b = q1.bits(); n = W.numel()
                    row[f"L{L}_g{g}_s{share}"] = dict(idx_bpw=b["idx"] / n, scale_bpw={k2: v / n for k2, v in b["scale"].items()})
        out["linear"].append(row)
    for sh, k in rem:
        x = cache[k]
        if x.dtype in (torch.bfloat16, torch.float16):
            raw = x.view(torch.int16).flatten().long() & 0xFFFF; ex = (raw >> 7) & 255 if x.dtype == torch.bfloat16 else (raw >> 10) & 31
            nb_m = 7 if x.dtype == torch.bfloat16 else 10; out["remainder"].append(dict(t=k, n=x.numel(), dtype=str(x.dtype), exp_H0=H0_bits(ex) / x.numel(), bits_if_exp_coded=1 + nb_m + H0_bits(ex) / x.numel()))
    res.update(out); os.makedirs("results", exist_ok=True)
    o = a.out or os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "stats_" + a.model.replace("/", "_") + ".json"); os.makedirs(os.path.dirname(o), exist_ok=True)
    json.dump(res, open(o, "w"), indent=1); print("wrote", o)
if __name__ == "__main__": main()
