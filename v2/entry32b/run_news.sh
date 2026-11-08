#!/bin/bash
cd /opt/dlami/nvme/e32; export HF_HOME=/opt/dlami/nvme/data/hf PATH=/opt/dlami/nvme/work/venv/bin:$PATH VLLM_UTIL=0.85
n=$1
python q32_local.py cfgs_x04n.json $n > run_$n.log 2>&1
python vreal_news.py /opt/dlami/nvme/tmp/q_$n /opt/dlami/nvme/tmp/real_${n}_sel.json /opt/dlami/nvme/e32/news_sel.jsonl > vreal_$n.log 2>&1
rm -rf /opt/dlami/nvme/tmp/q_$n
