#!/bin/bash
# Artifact track: Qwen3-8B quantize -> verify -> dev smoke -> final_proof (netns isolation, falls back to auto) -> bf16 export. Public HF model, no token.
set -x; cd "$(dirname "$0")"; HERE=$PWD; mkdir -p results; W=/workspace/w; mkdir -p $W
M=Qwen/Qwen3-8B
pip install -q torch==2.5.1 transformers==4.56.2 accelerate numpy safetensors constriction==0.5.0 huggingface_hub pytest 2>&1 | tail -2
export HF_HOME=/workspace/hf; export PYTHONPATH=$HERE/v2/harness/v3
SNAP=$(python -c "from huggingface_hub import snapshot_download as s;print(s('$M',allow_patterns=['*.json','*.safetensors','*.txt']))")
T1=$HERE/v2/tournament/T1/data
cd $HERE/v2/artifact
[ -f $W/art/model.tpq ] || { python quantize.py --model $SNAP --out $W/art --levels attn=3,o=3,mlp=3,down=3,emb=3,head=3 --gptq --calib $T1/s1_calib.jsonl --ncal 128 --sim-dir $W/sim && python writer.py assemble --art $W/art; }
python - <<P
import sys,json,math; sys.path[:0]=["$HERE/v2/artifact","$HERE/v2/harness/v3"]
import checks
arch=checks.arch_of_dir("$SNAP"); fp16=sum(2*math.prod(s) for _,s in arch.values())
json.dump(dict(base="$M",arch=arch,base_fp16_bytes=fp16,config=json.load(open("$SNAP/config.json"))),open("$W/art.base_arch.json","w"))
P
python verify.py --art $W/art --sim-dir $W/sim --sim-hashes $W/art.sim_hashes.json --base-arch $W/art.base_arch.json --out results/verify.json --workers 16
cd $HERE/v2/harness/v3
[ -f $W/data/exam.jsonl ] || python make_dev_smoke.py --base $M --out $W/data --per-cell 20 --pool 500 --n-ann 12 --n-teach 30 --n-nb 30
ISO=netns; unshare -n true 2>/dev/null || { echo "unshare -n unavailable -> isolation auto"; ISO=auto; }
mkdir -p results/run
python final_proof.py --exam $W/data/exam.jsonl --artifact $W/art --base-id $M --base-arch $W/art.base_arch.json --annotations $W/data/ann.jsonl --teach $W/data/teach.jsonl --neighbor $W/data/nb.jsonl --tag dev8b --out-dir $HERE/results/run --source-map S3=S2 --isolation $ISO > $HERE/results/final_proof.out 2>&1
echo "isolation=$ISO" > $HERE/results/ISO.txt
cp $W/art.quant_report.json $W/art.sim_hashes.json $HERE/results/ 2>/dev/null
touch $HERE/results/DONE
