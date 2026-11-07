#!/usr/bin/env bash
# Reference implementations (NOT run by the research agent). Run on a GPU box, not the Mac.
set -e
mkdir -p refs && cd refs
# 1. QTIP trellis quantizer (GPL-3.0). quantize_llama/quantize_finetune_llama.py; use NO-finetune settings (see REPORT sec 2)
git clone https://github.com/Cornell-RelaxML/qtip
(cd qtip && pip install -r requirements.txt && cd qtip-kernels && python setup.py install)
# 2. REAP expert pruning (Apache-2.0). MoE bases only (Qwen3-30B-A3B, GLM-4.5-Air, ...)
git clone https://github.com/CerebrasResearch/reap
(cd reap && bash scripts/build.sh)
# example (25% experts pruned, REAP criterion, seed 42); swap the dataset for OUR tool-call calibration set (never the locked exam):
# bash experiments/pruning-cli.sh 0 Qwen/Qwen3-30B-A3B reap 42 0.50 <calib_dataset> true true true false false
# 3. Other references (licences from memory, verify): 
git clone https://github.com/microsoft/VPTQ
git clone https://github.com/Vahe1994/AQLM
git clone https://github.com/IST-DASLab/SLQ          # statistically-lossless layerwise bit search (arXiv 2605.02404)
git clone https://github.com/jiwonsong-dev/SLEB       # block removal
git clone https://github.com/spcl/QuaRot              # fixed Hadamard (no training)
