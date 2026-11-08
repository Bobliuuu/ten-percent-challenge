#!/bin/bash
set -x; cd "$(dirname "$0")"; mkdir -p results
pip install -q "transformers==4.55.0" accelerate numpy huggingface_hub 2>&1 | tail -2
if [ "$1" = "32bb" ]; then
  T3_ITEMS=items_r2.jsonl T3_CALIB=calib_r2.jsonl T3_TAG=_r2 python code/runner.py 14
  T3_ITEMS=items_r3.jsonl T3_CALIB=calib_r3.jsonl T3_TAG=_r3 python code/runner.py 15
else
  python code/runner.py "$1"
fi
