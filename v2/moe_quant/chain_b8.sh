#!/bin/bash
cd "$(dirname "$0")"; L=$(python3 -c "import json;s=json.load(open('shards_b8.json'));print(' '.join(s['$2']))")
./vast_ssh.sh $1 "cd /root/code && (nohup bash -c 'sleep 5; while pgrep -f \"[b]ox_run.py\" >/dev/null; do sleep 10; done; python box_run.py cfgs_bank8.json base $L > /r/run_b8.log 2>&1' > /dev/null 2>&1 < /dev/null &); echo queued" 2>&1 | LC_ALL=C sed 's/@[0-9.]*//'
