export PATH=$HOME/v2env/bin:/usr/local/cuda-13.2/bin:$PATH CUDA_HOME=/usr/local/cuda-13.2
cd ~/work
while ! grep -aq FB_SWEEP_DONE run_fb2.log || ! grep -aq FREQ_BUILT chain_freq.log; do sleep 20; done
python probe2_box.py /home/ubuntu/work/loaded_k16_f165k cand_k16_f165k_ml ml_only_items.jsonl > p2_f165.log 2>&1
python score_ml_box.py out2/cand_k16_f165k_ml.json >> score.log 2>&1
echo F165_DONE
