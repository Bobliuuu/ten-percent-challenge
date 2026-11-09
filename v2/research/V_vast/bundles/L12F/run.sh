#!/bin/bash
set -x; cd "$(dirname "$0")"; mkdir -p results; export HF_HOME=/workspace/hf
pip uninstall -y -q torchvision torchaudio 2>&1 | tail -1
pip install -q torch==2.10.0 transformers==5.22.2 safetensors scipy huggingface_hub accelerate numpy 2>&1 | tail -2
python run_fbo.py --cfg best.json --out results/o --root fin --nlay 8 --margin 32 --lams ${1:-1} --draws 3 --export ${2:-1} > results/run_F.log 2>&1
touch results/DONE
