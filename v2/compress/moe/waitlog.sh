#!/bin/bash
# usage: waitlog.sh APPID PATTERN TIMEOUT_S ; prints matching lines when found
end=$((SECONDS+$3))
while [ $SECONDS -lt $end ]; do
  out=$(timeout 60 modal app logs $1 2>&1 | grep -E "$2" | tail -5)
  if [ -n "$out" ]; then echo "$out"; exit 0; fi
  sleep 20
done
echo timeout
