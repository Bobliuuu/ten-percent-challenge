#!/bin/bash
# usage: run_after.sh MAXRUNNERS "TASK:JOB ..." : waits until fewer than MAXRUNNERS runner.py processes have been alive for 3 consecutive 40s checks, then run_p.sh
cd "$(dirname "$0")"
ok=0
while [ $ok -lt 3 ]; do
  n=$(pgrep -fc "runner.p[y]"); if [ "$n" -lt "$1" ]; then ok=$((ok+1)); else ok=0; fi; sleep 40
done
bash run_p.sh "$2"
