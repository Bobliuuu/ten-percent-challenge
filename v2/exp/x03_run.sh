#!/bin/bash
# Bundle entry (queue worker): setup once, then run jobs listed in jobs.txt (one cfg name per line, e.g. A0_s11); new lines/cfgs may be pushed later.
# Exits when STOP exists and no job is left, or after 40 idle minutes.
cd "$(dirname "$0")"; mkdir -p results v2/exp/results/X03 /workspace/packs
exec > /workspace/worker.log 2>&1; set -x
pip install -q "vllm==0.11.0" "transformers==4.57.1" scipy "huggingface_hub<1.0" "datasets<4" safetensors accelerate 2>&1 | tail -2
export HF_HOME=/workspace/hf PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for i in 1 2 3 4 5; do python -c "from huggingface_hub import snapshot_download as s; s('Qwen/Qwen2.5-14B-Instruct', allow_patterns=['*.json','*.safetensors','*.txt'])" && break; sleep 30; done
idle=0
while true; do
  J=""
  for j in $(cat jobs.txt 2>/dev/null); do
    if [ ! -f "v2/exp/results/X03/${j}.DONE" ] && [ ! -f "v2/exp/results/X03/${j}.FAILED" ] && [ -f "v2/exp/cfgs/${j}.json" ]; then J=$j; break; fi
  done
  if [ -z "$J" ]; then
    if [ -f STOP ]; then break; fi
    sleep 20; idle=$((idle+20)); if [ $idle -gt 2400 ]; then break; fi; continue
  fi
  idle=0
  if grep -q '"mode": "sens"' v2/exp/cfgs/$J.json; then
    (cd v2/exp && python q14.py cfgs/$J.json $(python -c "import json;print(json.load(open('cfgs/$J.json'))['name'])") > results/X03/$J.log 2>&1) && touch v2/exp/results/X03/${J}.DONE || touch v2/exp/results/X03/${J}.FAILED
  else
    SETS=$(python -c "import json;print(json.load(open('v2/exp/cfgs/$J.json')).get('sets','SEL,GEN'))"); python v2/exp/run_retention.py --cfg v2/exp/cfgs/$J.json --stage all --sets $SETS --cleanup > v2/exp/results/X03/$J.log 2>&1 && touch v2/exp/results/X03/${J}.DONE || touch v2/exp/results/X03/${J}.FAILED
  fi
  rm -rf /workspace/packs/${J}_hf; df -h /workspace | tail -1
  cp -r v2/exp/results/X03/. results/
done
cp -r v2/exp/results/X03/. results/; cp /workspace/worker.log results/worker.log; touch results/DONE
