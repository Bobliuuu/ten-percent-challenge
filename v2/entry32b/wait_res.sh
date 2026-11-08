#!/bin/bash
# usage: wait_res.sh NAME... ; waits until all /res/NAME.json exist on volume tenpct-entry32b
for i in $(seq 1 400); do
  ok=1; for n in "$@"; do modal volume ls tenpct-entry32b /res 2>/dev/null | grep -q "res/$n.json" || ok=0; done
  [ $ok = 1 ] && { echo done; exit 0; }; sleep 120
done; echo timeout
