#!/bin/bash
set -x; cd "$(dirname "$0")"; mkdir -p results; pip install -q "transformers==4.55.0" accelerate numpy huggingface_hub zstandard safetensors 2>&1 | tail -2
cd v2/byte_tricks
python stats_weights.py prism-ml/Ternary-Bonsai-27B-unpacked --shards 1 --skip 2 --tensors 16 --discrete --out results/stats_ternary_skip2.json 2>&1 | tail -3
for m in Qwen/Qwen2.5-14B-Instruct Qwen/Qwen2.5-32B-Instruct Qwen/Qwen3-30B-A3B prism-ml/Bonsai-8B-unpacked prism-ml/Ternary-Bonsai-27B-unpacked; do python stats_embed.py $m 2>&1 | tail -2; done
cp results/stats_* ../../results/; touch ../../results/DONE
