#!/usr/bin/env bash
# Reproduce (run from repo root; needs python3.12, torch, numpy, transformers, datasets; HF access to textattack/bert-base-uncased-SST-2)
set -euo pipefail
python build.py --requantize                 # base -> compress/ (GPTQ+pack) -> artifact/{predict.py,model.qpk}; logs bytes
python feedback.py                           # mem0 (anchors only) / mem1 (+ feedback) from the SST-2 *train* feedback split
python harness/size_check.py artifact --memory mem/mem1.bin
python verify.py --memory mem/mem1.bin       # sandboxed no-base/no-network proof
# stand-in exam (dev split of SST-2 train). The real exam: python harness/lock.py FIRST (before building), then final_exam.py once per stage.
python harness/lock.py --split dev --out standin/LOCK_dev.json
python harness/final_exam.py --artifact artifact --memory mem/mem0.bin --stage 1 --split dev --lock standin/LOCK_dev.json --receipt-dir standin/receipts
python harness/final_exam.py --artifact artifact --memory mem/mem1.bin --stage 2 --split dev --lock standin/LOCK_dev.json --receipt-dir standin/receipts
