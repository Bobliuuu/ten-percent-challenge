#!/bin/bash
cd /workspace/bundle; mkdir -p results; pip install -q transformers==4.51.3 accelerate sentencepiece 2>&1 | tail -1
HF_HOME=/workspace/hf python run_abl.py cfg.json results/out > results/run.log 2>&1; touch results/DONE
