#!/bin/bash
# on a fresh box with /workspace/tp synced: install, download Bonsai, pack, assemble art, build extra data
set -ex; cd /workspace; export HF_HOME=/workspace/hf
pip install -q transformers==4.56.2 datasets accelerate huggingface_hub safetensors
SNAP=$(python -c "from huggingface_hub import snapshot_download as s;print(s('prism-ml/Bonsai-8B-unpacked'))")
python tp/v2/research/B_native/scripts/pack1.py "$SNAP" /workspace/art
for f in tokenizer.json tokenizer_config.json vocab.json merges.txt config.json generation_config.json; do cp $SNAP/$f /workspace/art/; done
cp tp/v2/bonsai/ship/pack_extra/* /workspace/art/; cp tp/v2/bonsai/ship/{answer.py,feedback.py,common.py,bonsai_loader.py,news_task.json} /workspace/art/; cp tp/v2/lanes/L1/feedback_calib.jsonl /workspace/art/   # SKILL7 calibration (182 unlabeled prompts); the NEWS one is v2/bonsai/data_news/feedback_calib.jsonl (use that only for news_task.json)
[ -n "$SKIP_DATA" ] || (cd tp/v2/bonsai/fbx && python build_data.py --out /workspace/extra)
mkdir -p /workspace/res; echo SETUPDONE
