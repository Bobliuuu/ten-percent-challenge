#!/bin/bash
# waits for sweeps then launches finals on box 1
cd ~/ten-percent/v2/bonsai/l1plus
until timeout 40 ./sx.sh 1 'tail -c 300 /workspace/res/swA.log | grep -q DONE && tail -c 300 /workspace/res/swS.log | grep -q DONE'; do sleep 20; done
echo SWEEPSDONE
for n in 1 2; do timeout 30 ./sx.sh 1 "cd /workspace/tp/v2/bonsai/l1plus && cp /workspace/res/swA.json.basepreds.json /workspace/res/fin$n.json.basepreds.json && HF_HOME=/workspace/hf setsid nohup python run_l1p.py --art /workspace/art --extra /workspace/extra --out /workspace/res/fin$n.json --mode final --cfgs cfgs_final$n.json --ndraw 5 > /workspace/res/fin$n.log 2>&1 < /dev/null & echo launched$n"; done
