export PATH=$HOME/v2env/bin:/usr/local/cuda-13.2/bin:$PATH CUDA_HOME=/usr/local/cuda-13.2
cd ~/work
uv pip install -p ~/v2env/bin/python accelerate > acc.log 2>&1
grep -aq "^DONE" unf_184.log && python fixvis.py v_k16_n184k > /dev/null 2>&1
cp -n art_load.py /dev/null 2>&1
python art_build.py /home/ubuntu/work/art_k16_n184k 16 184000 > art184.log 2>&1
cp hadamard_min.json art_k16_n184k/hadamard.json; cp art_load.py art_k16_n184k/
python art_load.py art_k16_n184k /home/ubuntu/work/loaded_k16_n184k cuda > load184.log 2>&1
python mkprobe_dir.py loaded_k16_n184k v_k16_n184k > mk184.log 2>&1
grep -a "^ml\|^mgsm\|^mltox\|^code" /dev/null; python - <<P
import json
for l in open("ml_items.jsonl"): pass
P
cp ml_items.jsonl ml_only_items.jsonl
python probe2_box.py /home/ubuntu/work/loaded_k16_n184k cand_k16_n184k_ml ml_only_items.jsonl > p2_184.log 2>&1
python score_ml_box.py out2/cand_k16_n184k_ml.json >> score.log 2>&1
echo N184_DONE
