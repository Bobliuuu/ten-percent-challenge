#!/bin/bash
cd ~/ten-percent/v2/feedback_v3; mkdir -p results
rsync -az --include='*.json' --include='log_*.txt' --exclude='*' tenpct-g2:/opt/dlami/nvme/work/out/ results/
