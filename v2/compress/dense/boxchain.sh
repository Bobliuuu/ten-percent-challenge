#!/bin/bash
cd /opt/dlami/nvme/work/dense && source ../venv/bin/activate
export DENSE_HF=/opt/dlami/nvme/work/dense/hf DENSE_DATA=/opt/dlami/nvme/work/dense/dense_data.json HF_HOME=/opt/dlami/nvme/hf
python local_run.py box14.json 0 1 2 3 4 5 6 7 8 9 10 11 > ../out/box14.log 2>&1
DENSE_GPU=1 python local_run.py box32.json > ../out/box32.log 2>&1
