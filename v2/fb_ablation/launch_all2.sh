#!/bin/bash
cd ~/ten-percent/v2/fb_ablation
for J in "$@"; do nohup python3 launch_vast_l11.py L11_$J --bundle ~/ten-percent/v2/fb_ablation/bundles/L11_$J --results-dir ~/ten-percent/v2/fb_ablation/results/vast_$J --min-vram 78 --max-vram 100 --max-dph 1.6 --min-inet 300 --disk 80 --max-hours 3 --pull-every 120 > /tmp/l11_launch_$J.log 2>&1 &
sleep 100; done
