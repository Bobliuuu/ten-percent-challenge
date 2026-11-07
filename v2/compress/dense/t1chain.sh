#!/bin/bash
cd /opt/dlami/nvme/work/dense && source ../venv/bin/activate
export DENSE_HF=/opt/dlami/nvme/work/dense/hf DENSE_DATA=/opt/dlami/nvme/work/dense/dense_data.json HF_HOME=/opt/dlami/nvme/hf DENSE_T1=/opt/dlami/nvme/work/dense/t1
python local_run.py $1 > ../out/$1.log 2>&1
