"""I5: the ONE byte-accounting function. Real bytes = every shipped file under a pack directory (weights/index streams, scales, codebooks,
embedding/head, tokenizer, config, decoder code, calibration/feedback files). No exclusions; no estimates.
  python bytes.py PACK_DIR --base Qwen/Qwen2.5-32B-Instruct [--target 9.90] [--est-ratio 9.88]
Exit 1 if real ratio > target+tol or |est-real| > 0.2 pp. fp16 size of the START model = sum(numel)*2 from safetensors metadata (or --fp16-bytes)."""
import os, sys, json, argparse
FP16_PARAMS = {  # fallback (published parameter counts); the safetensors metadata is preferred and logged when it disagrees
    "Qwen/Qwen2.5-0.5B-Instruct": 494032768, "Qwen/Qwen2.5-14B-Instruct": 14770033664,
    "Qwen/Qwen2.5-32B-Instruct": 32763876352, "Qwen/Qwen2.5-72B-Instruct": 72706203648}
def dir_bytes(d):
    tot = 0; files = {}
    for r, _, fs in os.walk(d):
        for f in fs:
            p = os.path.join(r, f)
            if os.path.islink(p) and not os.path.exists(p): continue
            s = os.path.getsize(p); files[os.path.relpath(p, d)] = s; tot += s
    return tot, files
def fp16_bytes(base, override=None):
    if override: return int(override), "override"
    try:
        from huggingface_hub import get_safetensors_metadata
        m = get_safetensors_metadata(base); n = sum(m.parameter_count.values()); return 2 * n, "safetensors_metadata"
    except Exception:
        if base in FP16_PARAMS: return 2 * FP16_PARAMS[base], "table"
        raise
def account(pack, base, fp16=None, target=9.90, tol=0.02, est_ratio=None):
    real, files = dir_bytes(pack); f16, src = fp16_bytes(base, fp16); ratio = 100.0 * real / f16
    out = dict(pack=os.path.basename(pack.rstrip("/")), base=base, real_bytes=real, fp16_bytes=f16, fp16_src=src, ratio_pct=ratio, target_pct=target,
               within_target=ratio <= target + tol, n_files=len(files), largest=sorted(files.items(), key=lambda kv: -kv[1])[:5])
    if est_ratio is not None: out["ratio_est_pct"] = est_ratio; out["est_ok"] = abs(est_ratio - ratio) <= 0.2
    out["ok"] = out["within_target"] and out.get("est_ok", True)
    return out
if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("pack"); ap.add_argument("--base", required=True); ap.add_argument("--fp16-bytes"); ap.add_argument("--target", type=float, default=9.90)
    ap.add_argument("--est-ratio", type=float); a = ap.parse_args()
    r = account(a.pack, a.base, a.fp16_bytes, a.target, est_ratio=a.est_ratio); print(json.dumps(r, indent=1)); sys.exit(0 if r["ok"] else 1)
