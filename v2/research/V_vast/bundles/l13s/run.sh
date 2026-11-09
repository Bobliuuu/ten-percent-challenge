#!/bin/bash
set -x; cd "$(dirname "$0")"; mkdir -p results; pip install -q "transformers==4.55.0" accelerate numpy huggingface_hub zstandard safetensors 2>&1 | tail -2
cd v2/byte_tricks
python stats_weights.py Qwen/Qwen2.5-14B-Instruct --shards 1 --tensors 16 2>&1 | tail -3
python stats_weights.py Qwen/Qwen2.5-32B-Instruct --shards 1 --tensors 16 2>&1 | tail -3
python stats_weights.py Qwen/Qwen3-30B-A3B --shards 1 --tensors 24 2>&1 | tail -3
python stats_weights.py prism-ml/Bonsai-8B-unpacked --shards 1 --tensors 16 --discrete 2>&1 | tail -3
python stats_weights.py prism-ml/Ternary-Bonsai-27B-unpacked --shards 1 --tensors 16 --discrete 2>&1 | tail -3
cp results/stats_* ../../results/; touch ../../results/DONE
