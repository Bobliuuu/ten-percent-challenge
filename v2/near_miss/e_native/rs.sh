#!/bin/bash
# usage: rs.sh PORT HOST cmd...   (ssh to a tenpct vast box with the shared key)
KEY=~/ten-percent/v2/research/V_vast/.vast_ssh_key; P=$1; H=$2; shift 2
exec ssh -i $KEY -p $P -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR root@$H "$@"
