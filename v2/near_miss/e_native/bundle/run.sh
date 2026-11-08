#!/bin/bash
cd "$(dirname "$0")"; mkdir -p results results/logs
pip install -q "transformers==4.57.1" accelerate tokenizers safetensors huggingface_hub sentencepiece 2>&1 | tail -2
export HF_HOME=/workspace/hf
MODELS=${@:-"microsoft/bitnet-b1.58-2B-4T:bitnet2b tiiuae/Falcon-E-3B-Instruct:falcone3b tiiuae/Falcon-E-1B-Instruct:falcone1b tiiuae/Falcon3-10B-Instruct-1.58bit:falcon3_10b HF1BitLLM/Llama3-8B-1.58-100B-tokens:llama3_8b tiiuae/Falcon3-7B-Instruct-1.58bit:falcon3_7b"}
for mt in $MODELS; do m=${mt%%:*}; t=${mt##*:}
  python code/ne_probe.py $m $t > results/logs/$t.log 2>&1 || echo "FAILED $t" >> results/logs/failures.txt
  tail -3 results/logs/$t.log
done
touch results/DONE
