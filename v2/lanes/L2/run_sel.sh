#!/bin/bash
# usage: run_sel.sh JOBS.json OUT.jsonl  (on the box after boot.sh; waits for BOOT_DONE)
until grep -q BOOT_DONE /root/boot.log; do sleep 20; done
cd /root/work/v2/lanes/L2 && HF_HOME=/root/work/hf python fb_l2x.py "$1" "$2" --model /root/work/unf > "$2.log" 2>&1
echo RUN_DONE >> "$2.log"
