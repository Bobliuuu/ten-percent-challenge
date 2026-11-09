#!/bin/bash
# Deterministic rebuild of the shipped artifact from the PUBLIC release (prism-ml/Ternary-Bonsai-2-27B-mlx-2bit + Qwen tokenizer files) and sha256 check against SHA256SUMS.
# Verified 2026-11-08 on a fresh rented box: all 13 files matched. usage (from v2/): bash lanes/L2/rebuild_artifact.sh WORKDIR   (needs python torch, safetensors, huggingface_hub, numpy; ~10 GB disk)
set -e; W=${1:-$HOME/work}; mkdir -p $W; V=$(cd "$(dirname "$0")/../.." && pwd); cd $W; export HF_HOME=${HF_HOME:-$W/hf}
KEEP=$V/ternary27/keep_p100k_n165k.json python $V/ternary27/art_build.py art_k16_f165k 16 1
cp $V/ternary27/hadamard_min.json art_k16_f165k/hadamard.json; cp $V/ternary27/art_load.py art_k16_f165k/
(cd art_k16_f165k && sha256sum -c $V/lanes/L2/SHA256SUMS)
