#!/bin/bash
# wait_done.sh BOX 'pattern-of-logs' : returns when every matching log has DONE (or a Traceback)
cd ~/ten-percent/v2/bonsai/l1plus
while true; do
  r=$(timeout 40 ./sx.sh $1 "for f in /workspace/res/$2; do tail -c 400 \$f | grep -qE 'DONE|Traceback|Error' && echo d || echo r; done" | sort | uniq -c | tr '\n' ' ')
  case "$r" in *r*) sleep 120;; *) echo "$r"; break;; esac
done
