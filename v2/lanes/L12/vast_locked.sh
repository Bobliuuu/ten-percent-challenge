#!/bin/bash
# Vast fallback runner (used 2026-11-09 when the Modal A100 queue was >1 h): rebuild the packed artifact from the HF model (run_pack.py), assemble, verify sha256 vs MANIFEST.json, then locked exam + 6 feedback draws.
# usage on a box with torch>=2.8: bash vast_locked.sh /root/w   (repo subset unpacked there)
set -x; W=${1:-/root/w}; L=$W/v2/lanes/L12; export HF_HOME=/root/hf
pip install -q transformers==5.22.2 safetensors scipy accelerate huggingface_hub numpy
cd $L && python3 run_pack.py --out /root/out_pack --root $W > /root/pack.log 2>&1; tail -3 /root/pack.log
python3 e2e_prep.py --out /root --root $W --art /root/art --src /root/out_pack/packed --hub $HF_HOME/hub > /root/prep.log 2>&1; tail -2 /root/prep.log
mkdir -p /root/out_exam /root/out_fb
python3 e2e_locked.py --mode exam --out /root/out_exam --root $W --art /root/art > /root/exam.log 2>&1
python3 e2e_locked.py --mode fb --draws 100,101,102,103,104,105 --out /root/out_fb --root $W --art /root/art > /root/fb.log 2>&1
echo ALL_DONE > /root/ALL_DONE
