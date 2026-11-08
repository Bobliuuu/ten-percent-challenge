#!/bin/bash
# usage: t3_retry.sh label jobidx minvram maxdph ; retries creation until the account has room, then manages the job
cd "$(dirname "$0")"
until python3 t3_create.py "$1" "$3" "$4" >> logs/retry_$1.log 2>&1; do sleep 120; done
sleep 30; python3 t3_mgr.py "$1" "$2" >> logs/mgr_$1.log 2>&1
