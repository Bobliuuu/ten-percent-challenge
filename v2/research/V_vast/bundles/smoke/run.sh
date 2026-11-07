#!/bin/bash
mkdir -p results; nvidia-smi --query-gpu=name,memory.total --format=csv > results/gpu.txt
python -c "import torch;print(torch.__version__,torch.cuda.is_available())" > results/torch.txt
python - <<'P'
import urllib.request
urllib.request.urlretrieve("https://huggingface.co/Qwen/Qwen3-0.6B/resolve/main/config.json","results/config.json")
P
touch results/DONE
