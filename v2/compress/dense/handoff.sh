#!/bin/bash
cd /opt/dlami/nvme/work/dense
while [ ! -f hf/dense/res_Qwen3-14B_t096_drop4.json ]; do sleep 15; done
ps aux | grep "[b]oxchain.sh" | awk '{print $2}' | xargs -r kill
ps aux | grep "[l]ocal_run.py box14.json" | awk '{print $2}' | xargs -r kill
sleep 5
source ../venv/bin/activate
export DENSE_HF=/opt/dlami/nvme/work/dense/hf DENSE_DATA=/opt/dlami/nvme/work/dense/dense_data.json HF_HOME=/opt/dlami/nvme/hf DENSE_GPU=1
ps aux | grep "[b]oxchain2.sh" | awk '{print $2}' | xargs -r kill
ps aux | grep "[l]ocal_run.py box2.json" | awk '{print $2}' | xargs -r kill
sleep 5
python local_run.py box32.json > ../out/box32.log 2>&1
unset DENSE_GPU
export DENSE_HF=/opt/dlami/nvme/work/dense/hf2
nohup python local_run.py box2.json > ../out/box2b.log 2>&1 &
