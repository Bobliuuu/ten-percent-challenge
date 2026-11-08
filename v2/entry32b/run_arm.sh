#!/bin/bash
# usage: run_arm.sh CFG NAME...   (runs on a GPU box; quantise, then real-generation eval via vreal2, then delete model dir)
cd /opt/dlami/nvme/e32; export HF_HOME=/opt/dlami/nvme/data/hf PATH=/opt/dlami/nvme/work/venv/bin:$PATH VLLM_UTIL=0.85
cfg=$1; shift
for n in "$@"; do
  python q32_local.py $cfg $n > run_$n.log 2>&1
  [ -f /opt/dlami/nvme/tmp/real_${n}_inj.json ] || python vreal2.py /opt/dlami/nvme/tmp/q_$n /opt/dlami/nvme/tmp/real_${n}_inj.json > vreal_$n.log 2>&1
  rm -rf /opt/dlami/nvme/tmp/q_$n
done
