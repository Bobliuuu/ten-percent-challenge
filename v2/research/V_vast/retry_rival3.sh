#!/bin/bash
# retry launching a rival3 bundle until Vast accepts (account instance cap shared across lanes)
j=$1; cd "$(dirname "$0")"; export VAST_CPU_RAM=120
for i in $(seq 1 40); do
  python3 launch_vast.py rival3_$j --results-dir ~/ten-percent/v2/rival3/results/vast_$j --min-vram 80 --max-vram 100 --max-dph 2.2 --min-inet 500 --disk 200 --max-hours 6 --pull-every 180 > logs/rival3_$j.log 2>&1
  grep -q "could not create\|no offers" logs/rival3_$j.log || exit 0
  sleep 300
done
