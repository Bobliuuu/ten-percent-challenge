#!/usr/bin/env bash
# Fresh-clone setup for the L3 entry. usage: bash setup_fresh.sh skill7x|bank6x [workdir]
# Needs: python3.11+, a CUDA GPU with >= 48 GB, `modal` CLI logged in to the workspace that owns the private volume tenpct-l3-art (artifact is not in git: 5.9 GB).
set -euo pipefail
T=${1:?task skill7x|bank6x}; W=${2:-$PWD/l3_work}; HERE=$(cd "$(dirname "$0")" && pwd); mkdir -p "$W/$T"
python3 -m venv "$W/venv"; . "$W/venv/bin/activate"; pip -q install "torch==2.6.0" "transformers==4.57.1" numpy scipy "huggingface_hub<1.0" modal
modal volume get tenpct-l3-art "/$T" "$W/" --force
cp "$HERE/pkg_$T/task.json" "$HERE/pkg_$T/calib_prompts.jsonl" "$W/$T/"
python3 - "$HERE" "$W/$T" "$T" <<'P'
import sys, json, hashlib
here, art, t = sys.argv[1:]; man = json.load(open(f"{here}/MANIFEST.json"))["rebuilt_sha256"][t]
for f, v in man.items():
    h = hashlib.sha256(); fh = open(f"{art}/{f}", "rb")
    for ch in iter(lambda: fh.read(1 << 24), b""): h.update(ch)
    assert h.hexdigest() == v["sha256"], f"sha256 mismatch {f}"
print("sha256 OK for", len(man), "files")
P
printf 'set a timer for ten minutes\nwhat is my account balance\n' > "$W/demo_msgs.txt"
python3 "$HERE/answer.py" "$W/$T" "$W/demo_msgs.txt"
echo "feedback: python3 $HERE/feedback.py $W/$T annotations.jsonl   (then answer.py again)"
