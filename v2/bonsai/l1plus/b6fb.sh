#!/bin/bash
cd /workspace/tp/v2/bonsai/l1plus; SNAP=$(ls -d /workspace/hf/hub/models--prism-ml--Bonsai-8B-unpacked/snapshots/*)
for c in "$@"; do HF_HOME=/workspace/hf python run_bank6.py /workspace/art $SNAP /workspace/tp /workspace/res/b6fb_$c.json cfgb6_$c.json fb > /workspace/res/b6fb_$c.log 2>&1; done
