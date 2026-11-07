#!/bin/sh
# stand-in (C2 DEV, 100 rows) end-to-end with the C2 feedback artifact and the real Qwen2.5-7B-Instruct bf16 base
cd "$(dirname "$0")/.."
L=locks/LOCK_standin_c2_7b.json; ART=/Users/jhome/ten-percent-artifacts/artC2_fb_fixed
TENPCT_GPU=A100-40GB modal run --detach exam_remote.py --lock $L --tag c2fb --stage 1 --exam-file locks/standin_c2_dev100.jsonl --artifact $ART --cand-gpu L40S
TENPCT_GPU=A100-40GB modal run --detach exam_remote.py --lock $L --tag c2fb --stage 2 --artifact $ART --memory tests/mem_c2_stage2.json --cand-gpu L40S
echo ALLDONE
