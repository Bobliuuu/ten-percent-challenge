#!/bin/bash
set -x; cd "$(dirname "$0")"; mkdir -p results; export HF_HOME=/workspace/hf
pip uninstall -y -q torchvision torchaudio 2>&1 | tail -1
pip install -q torch==2.10.0 transformers==5.22.2 safetensors scipy huggingface_hub accelerate 2>&1 | tail -2
python -c "import torch,transformers;print(torch.__version__,transformers.__version__,torch.cuda.is_available());from transformers import Qwen3_5MoeForCausalLM" > results/env.log 2>&1
python l12_core.py --model Qwen/Qwen3.5-35B-A3B --cfgs cfgsQ35.json --data data --out results > results/run_C.log 2>&1
touch results/DONE
