export PATH=$HOME/v2env/bin:/usr/local/cuda-13.2/bin:$PATH CUDA_HOME=/usr/local/cuda-13.2
cd ~/work
python build_ml_box.py > build_ml.log 2>&1
cat devlarge.jsonl ml_items.jsonl > all_items.jsonl; wc -l all_items.jsonl
python unfold_box2.py 0 0 /home/ubuntu/work/tb2_hf > unf0.log 2>&1
python unfold_box2.py 16 140000 /home/ubuntu/work/v_k16_n140k > unf1.log 2>&1
python art_build.py /home/ubuntu/work/art_k16_n140k 16 140000 > art.log 2>&1
cp hadamard_min.json art_k16_n140k/hadamard.json; cp art_load.py art_k16_n140k/
python art_load.py art_k16_n140k /home/ubuntu/work/loaded_k16_n140k cuda > load.log 2>&1
python verify_art.py loaded_k16_n140k v_k16_n140k > verify.log 2>&1
python mkprobe_dir.py loaded_k16_n140k v_k16_n140k > mk.log 2>&1
for pair in "/home/ubuntu/work/tb2_hf base_t2b" "/home/ubuntu/work/loaded_k16_n140k cand_k16" "Qwen/Qwen3.8-27B parent_q38"; do set -- $pair; [ -f out2/$2.json ] || python probe2_box.py $1 $2 all_items.jsonl > p2_$2.log 2>&1; python score_ml_box.py out2/$2.json >> score.log 2>&1; done
echo ALL_DONE
