#!/bin/bash
# INJ new-label (Jailbreak), Qwen3-4B lda,cal2,cal. Public HF model, no token.
set -x; cd "$(dirname "$0")"; mkdir -p results
pip install -q "transformers==4.57.*" accelerate numpy huggingface_hub 2>&1 | tail -2
python code/run_inl_final.py --model Qwen/Qwen3-4B --methods lda,cal2,cal --ndraw 10 --nosens --data data --out results/res5/Qwen3-4B
touch results/DONE
