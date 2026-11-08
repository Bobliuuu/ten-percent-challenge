#!/bin/bash
# on the box: launch_remote.sh NAME MODE CFGFILE [extra args]  -> /workspace/res/NAME.json/.log (detached, own process group)
N=$1; MODE=$2; CF=$3; shift 3
cd /workspace/tp/v2/bonsai/fbx; [ -f /workspace/res/$N.json.basepreds.json ] || cp /workspace/res/sweep0.json.basepreds.json /workspace/res/$N.json.basepreds.json
HF_HOME=/workspace/hf setsid nohup python run_fbx.py --art /workspace/art --extra /workspace/extra --out /workspace/res/$N.json --mode $MODE --cfgs $CF "$@" > /workspace/res/$N.log 2>&1 < /dev/null &
echo launched $N
