#!/bin/bash
# helper: push a job json to the running L9b vast box queue (uses launcher key; no host printed). usage: vq.sh job.json | vq.sh --cmd "shell cmd" | vq.sh --pull
KEY=~/ten-percent/v2/research/V_vast/.vast_ssh_key
read H P < <(vastai show instances --raw | python3 -c "
import json,sys
for i in json.load(sys.stdin):
    if i.get('label')=='tenpct-l9b': print(i['ssh_host'],i['ssh_port'])")
SSH="ssh -i $KEY -p $P -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR root@$H"
case "$1" in
 --cmd) $SSH "$2";;
 --pull) rsync -az -e "ssh -i $KEY -p $P -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR" root@$H:"$2" "$3";;
 --push) rsync -az -e "ssh -i $KEY -p $P -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR" "$2" root@$H:"$3";;
 *) scp -i $KEY -P $P -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR "$1" root@$H:/workspace/queue/$(basename $1);;
esac
