#!/bin/bash
# usage: restart.sh ID KEY1B KEY2 : kill this box's runs, run phase-1b shard then phase-2 shard sequentially
cd "$(dirname "$0")"
A=$(python3 -c "import json;print(' '.join(json.load(open('shards1b.json'))['$2']))"); B=$(python3 -c "import json;print(' '.join(json.load(open('shards2.json'))['$3']))")
./vast_ssh.sh $1 "pkill -f '[w]hile pgrep'; pkill -f '[b]ox_run.py'; sleep 3; echo killed" 2>&1 | LC_ALL=C sed 's/@[0-9.]*//'
./vast_ssh.sh $1 "cd /root/code && (nohup bash -c 'python box_run.py cfgs_sweep.json $A > /r/run1b.log 2>&1; python box_run.py cfgs_sweep2.json base $B > /r/run2.log 2>&1' > /dev/null 2>&1 < /dev/null &); echo restarted" 2>&1 | LC_ALL=C sed 's/@[0-9.]*//'
