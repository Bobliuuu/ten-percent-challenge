#!/bin/bash
# Fresh-clone setup for L9b: fetch the packed artifact from OUR PRIVATE Modal volume (needs `modal token` of the owning workspace; nothing is public), copy the shipped code next to it,
# verify body.bin sha256 against MANIFEST.json, run a smoke request. usage: bash setup_fresh_clone.sh [OUTDIR=./art_s7_k102_e3]
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"; ART="${1:-$PWD/art_s7_k102_e3}"
pip install -q "torch>=2.8" "transformers==5.22.2" numpy accelerate sentencepiece protobuf tiktoken safetensors modal
mkdir -p "$ART"
modal volume get tenpct-l9b l9b/art_s7_k102_e3/body.bin "$ART/body.bin" --force
modal volume get tenpct-l9b l9b/art_s7_k102_e3/manifest.json "$ART/manifest.json" --force
for f in tokenizer.json tokenizer_config.json chat_template.jinja; do modal volume get tenpct-l9b "l9b/art_s7_k102_e3/$f" "$ART/$f" --force; done
for f in core.py qpack.py packed_loader.py answer.py feedback.py fbcore.py annot.py skill_task.json feedback_cal.jsonl; do cp "$HERE/$f" "$ART/$f"; done
python3 - "$ART" "$HERE/MANIFEST.json" <<'PY'
import hashlib, json, sys, os
art, man = sys.argv[1], json.load(open(sys.argv[2]))
h = hashlib.sha256()
with open(os.path.join(art, "body.bin"), "rb") as f:
    for b in iter(lambda: f.read(1 << 24), b""): h.update(b)
want = man["artifact_sha256"]["body_bin"]; assert h.hexdigest() == want, f"body.bin sha256 mismatch: {h.hexdigest()} != {want}"
print("body.bin sha256 OK", want)
PY
echo "smoke test (needs one GPU >= 40 GB):"
python3 "$ART/answer.py" "$ART" "wake me up at seven" "what is the weather in paris"
echo "feedback: python3 $ART/feedback.py --artifact $ART --ann ann.jsonl   (ann rows {\"text\",\"label\"}, <= 25)"
