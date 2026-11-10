#!/bin/bash
# L1 scale-carrier locked exam on Vast (Modal queue >15 min). Artifact = public HF Bobliu/tenpct-l1-bonsai8b-grow + shipped code from the repo.
set -x; W=$PWD/w; export HF_HOME=/root/hf; mkdir -p results
pip install -q transformers==4.56.2 accelerate huggingface_hub numpy
python3 - <<'P'
from huggingface_hub import snapshot_download
import shutil, os
a = snapshot_download("Bobliu/tenpct-l1-bonsai8b-grow", local_dir="/root/art")
s = snapshot_download("prism-ml/Bonsai-8B-unpacked", revision="376f381570d6115bc03f82adcfa4af0c7672ae54", local_dir="/root/snap")
L = os.environ.get("W", "") 
P
L=$W/v2/lanes/L1
for f in answer.py feedback.py bonsai_loader.py common.py skill_task.json feedback_calib.jsonl tensor_manifest.json; do cmp -s $L/$f /root/art/$f || echo "DIFF vs HF artifact: $f"; done > results/art_diff.txt; cat results/art_diff.txt
for f in answer.py feedback.py bonsai_loader.py common.py skill_task.json feedback_calib.jsonl tensor_manifest.json; do cp $L/$f /root/art/$f; done
python3 -u $L/run_locked_scale.py /root/art /root/snap $W results/locked_L1_scale.json > results/locked_L1_scale.log 2>&1
touch results/DONE
