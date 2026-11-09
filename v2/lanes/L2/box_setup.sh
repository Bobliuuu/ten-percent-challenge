#!/bin/bash
# runs on the rented box: env + rebuild artifact from the released model + unfold. usage: box_setup.sh WORKDIR
set -e; W=${1:-$HOME/work}; mkdir -p $W; cd $W
export HF_HOME=$W/hf PYTHONUNBUFFERED=1
pip install -q transformers==5.22.2 safetensors scipy huggingface_hub accelerate datasets hf_transfer 2>&1 | tail -2
export HF_HUB_ENABLE_HF_TRANSFER=1
python -c "import torch,transformers;print(torch.__version__,transformers.__version__,torch.cuda.is_available())"
