#!/bin/bash
# builds Vast bundles under research/V_vast/bundles/l13{a,b,s}. Only dev/sel/cal data is copied (no exam/never/teach/locked files).
set -e; V2=~/ten-percent/v2; B=$V2/research/V_vast/bundles
for n in l13a l13b l13s; do
  rm -rf $B/$n; mkdir -p $B/$n/v2/byte_tricks $B/$n/v2/task_search/EASY $B/$n/v2/rival3/data/bank $B/$n/v2/rival3/data/clin $B/$n/results
  cp $V2/byte_tricks/*.py $V2/byte_tricks/arms_*.json $B/$n/v2/byte_tricks/
  cp $V2/task_search/EASY/{items_dev.jsonl,calib.jsonl,PLAN.json} $B/$n/v2/task_search/EASY/
  for s in bank clin; do cp $V2/rival3/data/$s/{sel.jsonl,cal.jsonl} $B/$n/v2/rival3/data/$s/; done
done
cat > $B/l13a/run.sh <<'X'
#!/bin/bash
set -x; cd "$(dirname "$0")"; mkdir -p results; export HF_HUB_ENABLE_HF_TRANSFER=0
pip install -q "transformers==4.55.0" accelerate numpy huggingface_hub zstandard 2>&1 | tail -2
for i in 1 2 3 4 5 6; do python -c "from huggingface_hub import snapshot_download as s; s('Qwen/Qwen2.5-14B-Instruct', allow_patterns=['*.safetensors','*.json','*.txt','*.jinja'])" && break; sleep 20; done
cd v2/byte_tricks && python run_arms.py --model Qwen/Qwen2.5-14B-Instruct --arms arms_full.json --out q14_skill7 --task SKILL7 --max_new 8 2>&1 | tee ../../results/run_arms.log
cp -r results/* ../../results/ 2>/dev/null; touch ../../results/DONE
X
sed 's/arms_full.json --out q14_skill7 --task SKILL7 --max_new 8/arms_sub.json --out q14_bank8 --task BANK8 --max_new 12/' $B/l13a/run.sh > $B/l13b/run.sh
cat > $B/l13s/run.sh <<'X'
#!/bin/bash
set -x; cd "$(dirname "$0")"; mkdir -p results; pip install -q "transformers==4.55.0" accelerate numpy huggingface_hub zstandard safetensors 2>&1 | tail -2
cd v2/byte_tricks
python stats_weights.py Qwen/Qwen2.5-14B-Instruct --shards 1 --tensors 16 2>&1 | tail -3
python stats_weights.py Qwen/Qwen2.5-32B-Instruct --shards 1 --tensors 16 2>&1 | tail -3
python stats_weights.py Qwen/Qwen3-30B-A3B --shards 1 --tensors 24 2>&1 | tail -3
python stats_weights.py prism-ml/Bonsai-8B-unpacked --shards 1 --tensors 16 --discrete 2>&1 | tail -3
python stats_weights.py prism-ml/Ternary-Bonsai-27B-unpacked --shards 1 --tensors 16 --discrete 2>&1 | tail -3
cp results/stats_* ../../results/; touch ../../results/DONE
X
# l13s2: ternary rerun (skip shard 0) + embed stats
rm -rf $B/l13s2; cp -r $B/l13s $B/l13s2; cp $V2/byte_tricks/stats_embed.py $B/l13s2/v2/byte_tricks/
cat > $B/l13s2/run.sh <<'X'
#!/bin/bash
set -x; cd "$(dirname "$0")"; mkdir -p results; pip install -q "transformers==4.55.0" accelerate numpy huggingface_hub zstandard safetensors 2>&1 | tail -2
cd v2/byte_tricks
python stats_weights.py prism-ml/Ternary-Bonsai-27B-unpacked --shards 1 --skip 2 --tensors 16 --discrete --out results/stats_ternary_skip2.json 2>&1 | tail -3
for m in Qwen/Qwen2.5-14B-Instruct Qwen/Qwen2.5-32B-Instruct Qwen/Qwen3-30B-A3B prism-ml/Bonsai-8B-unpacked prism-ml/Ternary-Bonsai-27B-unpacked; do python stats_embed.py $m 2>&1 | tail -2; done
cp results/stats_* ../../results/; touch ../../results/DONE
X
