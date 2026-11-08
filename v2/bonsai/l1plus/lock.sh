#!/bin/bash
cd ~/ten-percent/v2/bonsai/l1plus
until timeout 40 ./sx.sh 1 'tail -c 300 /workspace/res/fin1.log | grep -q DONE'; do sleep 20; done
echo FIN1DONE
timeout 30 ./sx.sh 1 "cd /workspace/tp/v2/bonsai/l1plus && cp /workspace/res/swA.json.basepreds.json /workspace/res/locked.json.basepreds.json && HF_HOME=/workspace/hf setsid nohup python run_l1p.py --art /workspace/art --extra /workspace/extra --out /workspace/res/locked.json --mode locked --cfgs cfgs_locked.json --ndraw 6 > /workspace/res/locked.log 2>&1 < /dev/null & echo launched"
