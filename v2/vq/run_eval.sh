#!/bin/bash
# usage: run_eval.sh NAME  -> vLLM eval of /tmp/q_NAME in a fresh process
cd /code && VLLM_UTIL=0.8 python vreal_both.py /tmp/q_$1 /vol/res/eval_$1.json > /root/eval_$1.log 2>&1
