#!/usr/bin/env python3
"""bytes_audit: sum EVERY shipped file of an artifact dir and print it as % of the base model's fp16 size (R1: size counts everything).

Usage:
  python3 v2/tools/bytes_audit.py ARTIFACT_DIR --base-params 14.77e9            # fp16 base = 2 * params
  python3 v2/tools/bytes_audit.py ARTIFACT_DIR --base-hf Qwen/Qwen2.5-14B-Instruct   # params from HF metadata (needs network + huggingface_hub)
  python3 v2/tools/bytes_audit.py ARTIFACT_DIR --base-fp16-bytes 29540000000
  options: --cap 10.0 (percent, default 10)  --json out.json  --compress (also measure zstd/xz of each category: what a container codec would save)
           --dups (hash all files + safetensors tensors; report identical files/tensors = dedup candidates)  --manifest MANIFEST.json (list of expected files; flags extra/missing)
Counts: regular files, symlink targets (flagged), hidden files, .git, __pycache__ (flagged; they ship if present). Directory entries are 0 bytes.
Categories: weights, scales/codebooks (names containing scale|codebook|lut|zero), tokenizer, config, code, template, data, other.
Exit code 0 if total <= cap % of base, 1 otherwise, 2 on usage error. Stdlib only (zstandard optional)."""
import argparse, hashlib, json, lzma, os, struct, sys, zlib

W_EXT = {".safetensors", ".bin", ".pt", ".pth", ".npz", ".npy", ".gguf", ".ckpt", ".h5", ".msgpack", ".ans", ".rans", ".pack"}
def category(rel):
    n = os.path.basename(rel).lower(); ext = os.path.splitext(n)[1]
    if any(k in n for k in ("scale", "codebook", "lut", "zeropoint", "zero_point")): return "scales/codebooks"
    if n.startswith("tokenizer") or n in ("vocab.json", "merges.txt", "vocab.txt", "added_tokens.json", "special_tokens_map.json") or ext in (".model", ".tiktoken"): return "tokenizer"
    if ext in W_EXT or n.endswith(".safetensors.index.json"): return "weights" if not n.endswith(".index.json") else "config"
    if ext in (".py", ".sh", ".c", ".cpp", ".cu", ".so", ".rs", ".js") or n == "makefile": return "code"
    if ext == ".jinja" or "chat_template" in n: return "template"
    if ext in (".json", ".yaml", ".yml", ".toml", ".cfg", ".ini") or n in ("license", "notice"): return "config"
    if ext in (".jsonl", ".csv", ".tsv", ".txt", ".md"): return "data/docs"
    return "other"

def walk(root):
    out = []; flags = []
    for dp, dns, fns in os.walk(root, followlinks=False):
        for fn in fns:
            p = os.path.join(dp, fn); rel = os.path.relpath(p, root)
            if os.path.islink(p):
                tgt = os.path.realpath(p)
                if os.path.isfile(tgt): out.append((rel, os.path.getsize(tgt))); flags.append(f"symlink counted at target size: {rel}")
                else: flags.append(f"dangling symlink: {rel}")
            else: out.append((rel, os.path.getsize(p)))
            if any(part.startswith(".") for part in rel.split(os.sep)): flags.append(f"hidden file/dir counted: {rel}")
            if "__pycache__" in rel or rel.endswith(".pyc"): flags.append(f"bytecode cache counted: {rel}")
    return sorted(out), flags

def st_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]; h = json.loads(f.read(n)); return n + 8, h
def tensor_hashes(path):
    hdr, h = st_header(path); res = {}
    with open(path, "rb") as f:
        for k, v in h.items():
            if k == "__metadata__": continue
            a, b = v["data_offsets"]; f.seek(hdr + a); res[k] = (hashlib.sha1(f.read(b - a)).hexdigest(), b - a)
    return res

def comp_sizes(blob):
    r = {"xz": len(lzma.compress(blob, preset=6)), "zlib": len(zlib.compress(blob, 9))}
    try:
        import zstandard; r["zstd19"] = len(zstandard.ZstdCompressor(level=19).compress(blob))
    except Exception: pass
    return r

def base_bytes(a):
    if a.base_fp16_bytes: return float(a.base_fp16_bytes)
    if a.base_params: return 2.0 * float(a.base_params)
    if a.base_hf:
        from huggingface_hub import HfApi
        info = HfApi().model_info(a.base_hf); tot = info.safetensors.total if info.safetensors else None
        if not tot: sys.exit("no safetensors metadata for " + a.base_hf)
        return 2.0 * tot
    sys.exit("give --base-params, --base-fp16-bytes or --base-hf")

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("dir"); ap.add_argument("--base-params"); ap.add_argument("--base-hf"); ap.add_argument("--base-fp16-bytes")
    ap.add_argument("--cap", type=float, default=10.0); ap.add_argument("--json"); ap.add_argument("--compress", action="store_true"); ap.add_argument("--dups", action="store_true"); ap.add_argument("--manifest")
    a = ap.parse_args()
    if not os.path.isdir(a.dir): sys.exit(2)
    base = base_bytes(a); files, flags = walk(a.dir); tot = sum(s for _, s in files)
    cats = {}
    for rel, s in files: cats.setdefault(category(rel), []).append((rel, s))
    print(f"artifact: {a.dir}\nbase fp16: {base/1e6:,.1f} MB   cap {a.cap}% = {base*a.cap/100/1e6:,.1f} MB")
    print(f"{'category':18s}{'files':>6s}{'bytes':>16s}{'% of base fp16':>16s}{'% of cap':>10s}" + ("   xz/zstd saving" if a.compress else ""))
    rows = []
    for c in sorted(cats, key=lambda c: -sum(s for _, s in cats[c])):
        b = sum(s for _, s in cats[c]); line = f"{c:18s}{len(cats[c]):6d}{b:16,d}{100*b/base:15.4f}%{100*b/(base*a.cap/100)*1.0:9.2f}%"
        r = dict(category=c, files=len(cats[c]), bytes=b, pct_base=100 * b / base)
        if a.compress and c != "weights":
            blob = b"".join(open(os.path.join(a.dir, rel), "rb").read() for rel, _ in cats[c]); cs = comp_sizes(blob); best = min(cs.values())
            line += f"   {b-best:,d} B ({100*(b-best)/base:.4f}% of base) via {min(cs, key=cs.get)}"; r["compressed"] = cs
        elif a.compress:   # weights: test the first 64 MB of the largest file only (entropy of the packed stream)
            rel = max(cats[c], key=lambda x: x[1])[0]; blob = open(os.path.join(a.dir, rel), "rb").read(64 << 20); cs = comp_sizes(blob)
            line += f"   (sample 64MB of {rel}: best codec keeps {100*min(cs.values())/len(blob):.1f}%)"; r["sample_ratio"] = min(cs.values()) / len(blob)
        rows.append(r); print(line)
    print(f"{'TOTAL':18s}{len(files):6d}{tot:16,d}{100*tot/base:15.4f}%{100*tot/(base*a.cap/100):9.2f}%")
    ok = tot <= base * a.cap / 100
    print(f"RESULT: {'PASS' if ok else 'FAIL'}  total = {100*tot/base:.4f}% of base fp16 (cap {a.cap}%), slack {(base*a.cap/100-tot)/1e6:,.1f} MB")
    print("largest files:"); [print(f"  {s:>16,d}  {100*s/base:7.4f}%  {rel}") for rel, s in sorted(files, key=lambda x: -x[1])[:8]]
    for f in flags: print("FLAG:", f)
    dup = []
    if a.dups:
        seen = {}
        for rel, s in files:
            p = os.path.join(a.dir, rel)
            if s > 0 and not rel.endswith(".safetensors"):
                h = hashlib.sha1(open(p, "rb").read()).hexdigest()
                if h in seen: dup.append(("file", rel, seen[h], s))
                else: seen[h] = rel
        ts = {}
        for rel, s in files:
            if rel.endswith(".safetensors"):
                for k, (h, n) in tensor_hashes(os.path.join(a.dir, rel)).items():
                    if h in ts and n > 1024: dup.append(("tensor", f"{rel}:{k}", ts[h], n))
                    else: ts[h] = f"{rel}:{k}"
        sv = sum(d[3] for d in dup); print(f"DEDUP: {len(dup)} identical files/tensors, {sv:,d} bytes = {100*sv/base:.4f}% of base fp16 removable")
        for d in dup[:10]: print("  ", d)
    if a.manifest:
        man = set(json.load(open(a.manifest)).get("files", [])) if os.path.isfile(a.manifest) else set(); have = {r for r, _ in files}
        print("MANIFEST extra (shipped, unlisted):", sorted(have - man)[:10], "missing:", sorted(man - have)[:10])
    if a.json: json.dump(dict(dir=a.dir, base_fp16_bytes=base, total_bytes=tot, pct_base=100 * tot / base, cap=a.cap, ok=ok, categories=rows, flags=flags, dup=dup), open(a.json, "w"), indent=1)
    sys.exit(0 if ok else 1)
if __name__ == "__main__": main()
