export PATH=$HOME/v2env/bin:/usr/local/cuda-13.2/bin:$PATH CUDA_HOME=/usr/local/cuda-13.2
cd ~/work
python tok_diag.py > tokdiag.log 2>&1
for pair in "/home/ubuntu/work/tb2_hf base_t2b" "/home/ubuntu/work/loaded_k16_n140k cand_k16" "Qwen/Qwen3.8-27B parent_q38"; do set -- $pair; [ -f out2/$2.json ] || python probe2_box.py $1 $2 all_items.jsonl > p2_$2.log 2>&1; python score_ml_box.py out2/$2.json >> score.log 2>&1; done
echo PROBES_DONE
