#!/bin/bash
set -x; cd "$(dirname "$0")"; mkdir -p results; export HF_HUB_ENABLE_HF_TRANSFER=0
pip install -q "transformers==4.55.0" accelerate numpy huggingface_hub zstandard 2>&1 | tail -2
for i in 1 2 3 4 5 6; do python -c "from huggingface_hub import snapshot_download as s; s('Qwen/Qwen2.5-14B-Instruct', allow_patterns=['*.safetensors','*.json','*.txt','*.jinja'])" && break; sleep 20; done
cd v2/byte_tricks && python run_arms.py --model Qwen/Qwen2.5-14B-Instruct --arms arms_full.json --out q14_skill7 --task SKILL7 --max_new 8 2>&1 | tee ../../results/run_arms.log
cp -r results/* ../../results/ 2>/dev/null; touch ../../results/DONE
