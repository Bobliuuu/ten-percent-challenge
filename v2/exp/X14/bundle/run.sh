#!/bin/bash
mkdir -p results
pip install -q vllm==0.11.0 "transformers==4.57.1" > results/pip.log 2>&1
export HF_HUB_ENABLE_HF_TRANSFER=0
python step1.py > results/step1.log 2>&1
touch results/DONE
