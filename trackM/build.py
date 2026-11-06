#!/usr/bin/env python
"""base -> artifact. Deterministic packaging; logs every byte.
  python build.py [--bin BIN | --requantize] [--out artifact] [--log build/build_log.json]
--bin BIN       packed model in the trackA container format (default: ../trackA/model.bin). Any bit allocation works (drop-in).
--requantize    first run the compression stage (../trackA/final.py: GPTQ mixed 2/3/4 bit + embedding split + position trim) to BIN.
Quantization (GPU/MPS float math) is the only non-bit-reproducible step; everything after is pure byte copying, hence the sha256s in the log."""
import argparse, hashlib, json, lzma, os, shutil, struct, subprocess, sys, zlib
H = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser(); ap.add_argument("--bin", default=os.path.join(H, "../trackA/model.bin")); ap.add_argument("--requantize", action="store_true")
ap.add_argument("--out", default=os.path.join(H, "artifact")); ap.add_argument("--log", default=os.path.join(H, "build/build_log.json"))
a = ap.parse_args()
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
if a.requantize:
    a.bin = os.path.join(H, "build/model_requant.bin"); os.makedirs(os.path.dirname(a.bin), exist_ok=True)
    subprocess.run([sys.executable, os.path.join(H, "../trackA/final.py"), a.bin], check=True)
os.makedirs(a.out, exist_ok=True)
for f in os.listdir(a.out):
    if f not in ("predict.py",): shutil.rmtree(os.path.join(a.out, f)) if os.path.isdir(os.path.join(a.out, f)) else os.remove(os.path.join(a.out, f))   # artifact = exactly predict.py + model.qpk
shutil.copyfile(a.bin, os.path.join(a.out, "model.qpk"))
raw = open(a.bin, "rb").read(); n = struct.unpack("<I", raw[:4])[0]; meta = json.loads(raw[4:4 + n])
parts = {"container_header": 4 + n, "vocab_zlib": meta["vocab_len"], "vocab_keep_mask": (meta["cfg"]["vocab_size"] + 7) // 8}
for k, d in meta["q"].items():
    O, I = d["shape"]; ng = O * (I // (I if d["g"] <= 0 else d["g"]))
    key = "embeddings" if "emb" in k else "pooler" if "pooler" in k else "encoder_linear"
    parts[key + "_packed"] = parts.get(key + "_packed", 0) + (O * I * d["bits"] + 7) // 8
    parts[key + "_scales_zeros_fp16"] = parts.get(key + "_scales_zeros_fp16", 0) + 4 * ng
parts["small_tensors_fp16"] = sum(2 * eval("*".join(map(str, d["shape"]))) for d in meta["fp16"].values())
base_bytes = 437958644
log = dict(source_bin=os.path.abspath(a.bin), model_bin_sha256=sha(a.bin), predict_py_sha256=sha(os.path.join(a.out, "predict.py")),
           model_bin_parts_bytes=parts, sum_parts=sum(parts.values()),
           files={f: os.path.getsize(os.path.join(a.out, f)) for f in sorted(os.listdir(a.out))}, base_fp32_bytes=base_bytes, budget_bytes=int(base_bytes * 0.10))
log["artifact_total_bytes"] = sum(log["files"].values()); log["ratio_pct"] = 100 * log["artifact_total_bytes"] / base_bytes
log["headroom_for_memory_bytes"] = log["budget_bytes"] - log["artifact_total_bytes"]
log["info_lzma_whole_bin_bytes"] = len(lzma.compress(raw, preset=9 | lzma.PRESET_EXTREME))   # potential entropy-coding gain (NOT applied)
os.makedirs(os.path.dirname(a.log), exist_ok=True); json.dump(log, open(a.log, "w"), indent=1)
print(json.dumps(log, indent=1))
