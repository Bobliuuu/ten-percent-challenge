#!/bin/bash
# annotation-aware requantization feedback, Qwen3-8B newlabel only (natural run finishes on AWS). Public HF model, no token. ~40GB VRAM, ~30 min.
set -x; cd "$(dirname "$0")"; mkdir -p results; cd code
pip install -q "transformers==4.57.*" accelerate numpy huggingface_hub 2>&1 | tail -2
python fbq4.py --model Qwen/Qwen3-8B --scenario newlabel --ndraw 8 --k 1 --variants B0,V2,V3 --out ../results/nl_8B_k1.json > ../results/nl_8B_k1.log 2>&1
touch ../results/DONE
