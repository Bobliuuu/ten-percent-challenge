#!/bin/bash
# DEV-ONLY harness v3 (label-mode copy) final_proof for the Bonsai NEWS artifact. Runs on the A10G box. Never uses a locked file.
cd /opt/dlami/nvme/w/v2/bonsai/h3 && . /opt/dlami/nvme/v/bin/activate
export BONSAI_SHUFFLE_POOL=/opt/dlami/nvme/w/data_news/shufpool.jsonl HF_HOME=/opt/dlami/nvme/hf TMPDIR=/opt/dlami/nvme/tmp PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True LABEL_CHANCE=0.25
W=/opt/dlami/nvme/w
python final_proof.py --exam $W/data_news/exam.jsonl --artifact $W/artH --base-id prism-ml/Bonsai-8B-unpacked --base-arch $W/artH.base_arch.json \
  --annotations $W/data_news/ann.jsonl --teach $W/data_news/teach.jsonl --neighbor $W/data_news/nb.jsonl --tag bonsai_news_dev --out-dir $W/run_news \
  --source-map S2probe=S2,S1=S1,S3=S3 --python /opt/dlami/nvme/v/bin/python > $W/proof_news.log 2>&1
