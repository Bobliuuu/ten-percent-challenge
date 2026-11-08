#!/bin/bash
# usage: run_p.sh "TASK:JOB TASK:JOB ..." (sequential)
set -x; cd "$(dirname "$0")"; mkdir -p results
pip install -q "transformers==4.55.0" accelerate numpy huggingface_hub 2>&1 | tail -2
for tj in $1; do t=${tj%%:*}; j=${tj##*:}
  T3_ITEMS=items_p_$t.jsonl T3_CALIB=calib_p_$t.jsonl T3_TAG=_p_$t python code/runner.py $j
done
