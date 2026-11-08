#!/bin/bash
# pull.sh 1|2 : rsync result files of a box into v2/bonsai/l1plus/results/ (hosts from ~/.l1p_boxes, never committed). basepreds files are skipped.
. ~/.l1p_boxes; N=$1; eval "set -- \$B$N"; K=$HOME/ten-percent/v2/research/V_vast/.vast_ssh_key; D=$HOME/ten-percent/v2/bonsai/l1plus/results; mkdir -p $D
rsync -az -e "ssh -i $K -p $2 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR" --exclude '*basepreds*' --exclude smoke* root@$1:/workspace/res/ $D/ 2>/dev/null
