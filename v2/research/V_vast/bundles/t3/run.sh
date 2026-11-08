#!/bin/bash
set -x; cd "$(dirname "$0")"; mkdir -p results
pip install -q "transformers==4.55.0" accelerate numpy huggingface_hub 2>&1 | tail -2
python code/runner.py "$1"
