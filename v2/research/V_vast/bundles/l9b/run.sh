#!/bin/bash
# L9b queue runner: runs jobs (json files) dropped into /workspace/queue sequentially via runjob.main; STOP file ends it. Results (small json) copied to results/.
set -x; cd "$(dirname "$0")"; HERE=$PWD; mkdir -p results /workspace/queue /workspace/queue/done /workspace/r/l9b/res
pip install -q transformers==5.22.2 accelerate numpy huggingface_hub sentencepiece protobuf tiktoken safetensors kernels 2>&1 | tail -2
ln -sfn /workspace/r /r; mkdir -p /root; rm -rf /root/l9b; ln -sfn $HERE/l9b /root/l9b

export HF_HOME=/workspace/hf HF_HUB_ENABLE_HF_TRANSFER=0
while true; do
  J=$(ls /workspace/queue/*.json 2>/dev/null | head -1)
  if [ -n "$J" ]; then
    echo "RUN $J"; python -c "
import sys,json; sys.path.insert(0,'/root/l9b'); import runjob; runjob.main(json.load(open('$J')))" >> /workspace/queue/run_$(basename $J).log 2>&1
    echo "exit $?"; mv $J /workspace/queue/done/; cp /workspace/queue/run_*.log results/ 2>/dev/null; cp /r/l9b/res/*.json results/ 2>/dev/null
  elif [ -f /workspace/queue/STOP ]; then touch results/DONE; break
  else sleep 20; fi
done
