#!/bin/bash
# pull results from the box every 15 min until 11:20Z
D=~/ten-percent/v2/combo/TOOLSEL
while [ "$(date -u +%H%M)" \< "1120" ] || [ "$(date -u +%H)" \< "05" ]; do
  rsync -a tenpct-g17:combo/TOOLSEL/results/ $D/results/ 2>/dev/null
  rsync -a --include='log_*.txt' --exclude='*' tenpct-g17:combo/TOOLSEL/ $D/box_logs/ 2>/dev/null
  sleep 900
done
