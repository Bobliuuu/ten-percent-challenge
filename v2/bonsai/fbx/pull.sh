#!/bin/bash
# pull.sh : rsync results of both boxes into v2/bonsai/fbx/results/box{1,2}   (hosts come from ~/.fbx_boxes, never committed)
D=$HOME/ten-percent/v2/bonsai/fbx; K=$HOME/ten-percent/v2/research/V_vast/.vast_ssh_key; . ~/.fbx_boxes
for n in 1 2; do eval "set -- \$B$n"; mkdir -p $D/results/box$n
rsync -az -e "ssh -i $K -p $2 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR" --exclude old root@$1:/workspace/res/ $D/results/box$n/ 2>/dev/null; done
