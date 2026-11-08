#!/bin/bash
cd ~/ten-percent/v2/bonsai/l1plus
until timeout 40 ./sx.sh 1 'tail -c 300 /workspace/res/cont.log | grep -q DONE'; do sleep 20; done
echo CONTDONE
timeout 30 ./sx.sh 1 "cd /workspace/tp/v2/bonsai/l1plus && for n in proxy robust; do cp /workspace/res/swA.json.basepreds.json /workspace/res/\$n.json.basepreds.json; done; (HF_HOME=/workspace/hf setsid nohup bash -c 'python run_l1p.py --art /workspace/art --extra /workspace/extra --out /workspace/res/proxy.json --mode proxy --cfgs cfgs_proxy.json --ndraw 3 > /workspace/res/proxy.log 2>&1; python run_l1p.py --art /workspace/art --extra /workspace/extra --out /workspace/res/robust.json --mode robust --cfgs cfgs_robust.json --ndraw 2 > /workspace/res/robust.log 2>&1' > /dev/null 2>&1 < /dev/null &) ; echo launched"
