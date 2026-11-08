#!/bin/bash
# on box 2: b6sweep.sh STREAM cfg... (sequential BANK6 dev runs)
cd /workspace/tp/v2/bonsai/l1plus; SNAP=$(ls -d /workspace/hf/hub/models--prism-ml--Bonsai-8B-unpacked/snapshots/*)
for c in "$@"; do HF_HOME=/workspace/hf python run_bank6.py /workspace/art $SNAP /workspace/tp /workspace/res/b6dev_$c.json cfgb6_$c.json dev > /workspace/res/b6dev_$c.log 2>&1; done
