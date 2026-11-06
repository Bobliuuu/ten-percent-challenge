#!/bin/sh
# stand-in (DEV 100 rows) end-to-end with the REAL altpush artifact and the REAL base (Qwen2.5-7B-Instruct bf16)
cd "$(dirname "$0")/.."
L=locks/LOCK_standin_mdc6_7b.json; ART=/Users/jhome/ten-percent-artifacts/art7b_c115
TENPCT_GPU=A100-40GB modal run --detach exam_remote.py --lock $L --tag art7b --stage 1 --exam-file locks/standin_mdc6_dev100.jsonl --artifact $ART --cand-gpu L40S
TENPCT_GPU=A100-40GB modal run --detach exam_remote.py --lock $L --tag art7b --stage 2 --artifact $ART --memory tests/mem_stage2.json --cand-gpu L40S
echo ALLDONE
