#!/bin/bash
cd ~/ten-percent/v2/research/V_vast
for J in "$@"; do nohup python3 launch_vast.py L11_$J --bundle ~/ten-percent/v2/fb_ablation/bundles/L11_$J --results-dir ~/ten-percent/v2/fb_ablation/results/$J --min-vram 78 --max-vram 100 --max-dph 1.6 --min-inet 300 --disk 80 --max-hours 4 --pull-every 120 > logs/L11_$J.log 2>&1 &
sleep 75; done
