#!/bin/bash
# usage: sx.sh HOST PORT cmd...   (ssh to the tenpct-fbx box)
H=$1; P=$2; shift 2
exec ssh -i $HOME/ten-percent/v2/research/V_vast/.vast_ssh_key -p $P -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -o ServerAliveInterval=30 root@$H "$@"
