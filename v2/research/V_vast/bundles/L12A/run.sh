#!/bin/bash
set -x; cd "$(dirname "$0")"; mkdir -p results; export HF_HOME=/workspace/hf
pip install -q transformers==5.22.2 safetensors scipy huggingface_hub accelerate 2>&1 | tail -2
python -c "import torch,transformers;print(torch.__version__,transformers.__version__)"
python l12_core.py --cfgs cfgsA.json --data data --out results > results/run_A.log 2>&1
touch results/DONE
