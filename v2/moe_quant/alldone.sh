#!/bin/bash
cd "$(dirname "$0")"
while true; do n=0; for id in 59721922 59721926 59721927; do c=$(./vast_ssh.sh $id "pgrep -f '[b]ox_run.py|[c]fgs_bank8|[c]fgs_proxy|[w]hile pgrep' | wc -l" 2>/dev/null); n=$((n+${c:-1})); done; [ "$n" = 0 ] && break; sleep 120; done; ./pull.sh; echo ALLDONE
