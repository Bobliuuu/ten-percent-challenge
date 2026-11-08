#!/bin/bash
cd "$(dirname "$0")"
./vast_cp.sh $1 up cfgs_proxy.json /root/code/cfgs_proxy.json
./vast_ssh.sh $1 "cd /root/code && (nohup bash -c 'sleep 8; while pgrep -f \"[c]fgs_bank8\" >/dev/null; do sleep 10; done; python box_r\"\"un.py cfgs_proxy.json proxy1 > /r/run_proxy.log 2>&1' > /dev/null 2>&1 < /dev/null &); echo queued" 2>&1 | LC_ALL=C sed 's/@[0-9.]*//'
