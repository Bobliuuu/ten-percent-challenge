#!/bin/bash
# sync.sh 1|2 : rsync the needed repo subset to /workspace/tp on the box (hosts in ~/.l1p_boxes, never committed)
. ~/.l1p_boxes; eval "set -- \$B$1"; H=$1; P=$2; R=$HOME/ten-percent
K=$R/v2/research/V_vast/.vast_ssh_key
ssh -i $K -p $P -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR root@$H "mkdir -p /workspace/tp/v2/moe_quant /workspace/tp/v2/research" </dev/null
cd $R && rsync -az --relative -e "ssh -i $K -p $P -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR" --exclude '__pycache__' --exclude 'results' v2/exam v2/bonsai v2/task_search v2/moe_quant/data v2/research/B_native v2/fb_edit root@$H:/workspace/tp/
