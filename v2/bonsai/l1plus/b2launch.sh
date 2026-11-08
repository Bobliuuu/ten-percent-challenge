#!/bin/bash
# waits for box 2 setup, then launches BANK6 dev (default cols config)
cd ~/ten-percent/v2/bonsai/l1plus
until timeout 40 ./sx.sh 2 'grep -q SETUPDONE /workspace/setup2.log'; do sleep 20; done
echo SETUP2DONE
timeout 60 ./sx.sh 2 'cp /workspace/tp/v2/bonsai/ship/answer.py /workspace/tp/v2/bonsai/ship/feedback.py /workspace/tp/v2/bonsai/ship/common.py /workspace/tp/v2/bonsai/ship/bonsai_loader.py /workspace/art/; echo "{\"tag\":\"cols_default\"}" > /workspace/cfg_default.json; SNAP=$(ls -d /workspace/hf/hub/models--prism-ml--Bonsai-8B-unpacked/snapshots/*); cd /workspace/tp/v2/bonsai/l1plus; HF_HOME=/workspace/hf setsid nohup python run_bank6.py /workspace/art $SNAP /workspace/tp /workspace/res/bank6_dev_default.json /workspace/cfg_default.json dev > /workspace/res/bank6_dev_default.log 2>&1 < /dev/null & echo launched'
