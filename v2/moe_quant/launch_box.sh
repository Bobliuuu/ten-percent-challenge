#!/bin/bash
# usage: launch_box.sh INSTANCE_ID SHARD_KEY   (runs base + shard configs in one process)
cd "$(dirname "$0")"; L=$(python3 -c "import json;s=json.load(open('shards.json'));print(' '.join(s['$2']))")
./vast_cp.sh $1 up cfgs_sweep.json /root/code/cfgs_sweep.json
./vast_ssh.sh $1 "cd /root/code && (nohup python box_run.py cfgs_sweep.json base $L > /r/run.log 2>&1 < /dev/null &) ; echo started" 2>&1 | LC_ALL=C sed 's/@[0-9.]*//'
