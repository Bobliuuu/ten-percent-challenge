#!/bin/bash
# run on a fresh vast box: installs deps and pre-downloads the model
pip install -q "transformers==4.57.1" accelerate hf_transfer huggingface_hub numpy 2>&1 | tail -2
mkdir -p /root/rep /root/out/dense
HF_HUB_ENABLE_HF_TRANSFER=1 python3 -c "from huggingface_hub import snapshot_download; snapshot_download(__import__('os').environ.get('MODEL','Qwen/Qwen2.5-32B-Instruct'), allow_patterns=['*.json','*.safetensors','*.txt'])" 2>&1 | tail -1
echo SETUP_DONE
