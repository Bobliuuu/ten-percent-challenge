#!/bin/bash
# sx.sh 1|2 'cmd'  (hosts in ~/.l1p_boxes)
. ~/.l1p_boxes; eval "set -- \$B$1 \"\$2\""; H=$1; P=$2; shift 2
exec ssh -i $HOME/ten-percent/v2/research/V_vast/.vast_ssh_key -p $P -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -o ServerAliveInterval=30 root@$H "$@" </dev/null
