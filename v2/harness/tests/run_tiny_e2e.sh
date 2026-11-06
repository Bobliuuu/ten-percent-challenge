#!/bin/sh
# end-to-end stand-in test on DEV: stage 1, stage 2 (memory changed), then a repeat of stage 1 and of stage 2 (must be REFUSED)
cd "$(dirname "$0")/.."
L=locks/LOCK_standin_mdc6_10pct.json
modal run --detach exam_remote.py --lock $L --tag tiny --stage 1 --exam-file locks/standin_mdc6_dev.jsonl --artifact tests/tiny
modal run --detach exam_remote.py --lock $L --tag tiny --stage 2 --artifact tests/tiny --memory tests/mem_stage2.json
echo "--- repeat stage 1 (expect REFUSED) ---"
modal run --detach exam_remote.py --lock $L --tag tiny --stage 1 --exam-file locks/standin_mdc6_dev.jsonl --artifact tests/tiny
echo "--- repeat stage 2 with local claim removed (expect REFUSED by Modal Dict) ---"
rm -f receipts/standin_tiny/stage2.claim
modal run --detach exam_remote.py --lock $L --tag tiny --stage 2 --artifact tests/tiny --memory tests/mem_stage2.json
echo ALLDONE
