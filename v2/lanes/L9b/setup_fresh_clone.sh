#!/bin/bash
# Fresh-clone setup for L9b: fetch the packed artifact from OUR PRIVATE Modal volume (needs `modal token` of the owning workspace; nothing is public), copy the shipped code next to it,
# verify body.bin sha256 against MANIFEST.json, run a smoke request. usage: bash setup_fresh_clone.sh [OUTDIR=./art_s7_k102_e3]
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"; ART="${1:-$PWD/art_s7_k102_e3}"
pip install -q "torch>=2.8" "transformers==5.22.2" numpy accelerate sentencepiece protobuf tiktoken safetensors $( [ -n "${L9B_SRC:-}" ] || echo modal )
mkdir -p "$ART"
fetch() { if [ -n "${L9B_SRC:-}" ]; then cp "$L9B_SRC/art_s7_k102_e3/$1" "$ART/$1"; else modal volume get tenpct-l9b "l9b/art_s7_k102_e3/$1" "$ART/$1" --force; fi; }   # L9B_SRC=<dir> (e.g. the volume mounted in a container) skips the modal CLI
for f in body.bin manifest.json tokenizer.json tokenizer_config.json chat_template.jinja; do fetch "$f"; done
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
