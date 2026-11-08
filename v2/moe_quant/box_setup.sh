#!/bin/bash
mkdir -p /r/res /r/hf /root/code; export HF_HOME=/r/hf
pip install -q torch==2.8.0 transformers==4.57.1 accelerate scipy numpy "huggingface_hub<1.0" 2>&1 | tail -2
python -c "
from huggingface_hub import snapshot_download
print(snapshot_download('Qwen/Qwen3-30B-A3B-Instruct-2507', allow_patterns=['*.json','*.safetensors','*.txt','*.model']))"
python -c "import torch;print(torch.__version__, torch.cuda.is_available(), torch.zeros(1).cuda()+1)"
echo SETUP_DONE
