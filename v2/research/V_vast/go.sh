#!/bin/bash
# Launch all unfinished jobs on Vast (3 instances at once). Usage: bash go.sh [artifact_iso INJ_q4b requant_8b]
cd "$(dirname "$0")"; mkdir -p logs; J=${@:-artifact_iso INJ_q4b requant_8b}
V2=~/ten-percent/v2
for j in $J; do case $j in
 artifact_iso) nohup python3 launch_vast.py artifact_iso --results-dir $V2/artifact/results/vast_iso --min-vram 24 --max-vram 49 --max-dph 0.8 --min-inet 500 --disk 150 --max-hours 3 --pull-every 120 > logs/artifact_iso.log 2>&1 & ;;
 INJ_q4b) nohup python3 launch_vast.py INJ_q4b --results-dir $V2/combo/INJ_newlabel/results_vast --min-vram 24 --max-vram 49 --max-dph 0.8 --min-inet 500 --disk 80 --max-hours 3 --pull-every 120 > logs/INJ_q4b.log 2>&1 & ;;
 requant_8b) nohup python3 launch_vast.py requant_8b --results-dir $V2/research/E_infra/fbq/results/vast --min-vram 46 --max-vram 100 --max-dph 1.2 --min-inet 500 --disk 100 --max-hours 3 --pull-every 120 > logs/requant_8b.log 2>&1 & ;;
esac; done
