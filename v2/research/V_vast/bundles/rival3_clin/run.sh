#!/bin/bash
pip install -q "transformers==4.57.1" accelerate scipy "huggingface_hub<1.0" numpy > /workspace/pip.log 2>&1
mkdir -p results
python3 vast_runner.py cfgs.json > results/run_stdout.log 2>&1
touch results/DONE
