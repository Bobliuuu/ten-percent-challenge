#!/bin/bash
cd /opt/dlami/nvme/work/dense && source ../venv/bin/activate
export DENSE_HF=/opt/dlami/nvme/work/dense/hf2 DENSE_DATA=/opt/dlami/nvme/work/dense/dense_data.json HF_HOME=/opt/dlami/nvme/hf
mkdir -p hf2/dense
python local_run.py box2.json > ../out/box2.log 2>&1
