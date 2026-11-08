# shared locked SKILL7 exam: base ONCE, candidate (f165k) ONCE, then Recipe feedback on candidate (<=25 ann from ann_pool; real/shuffled/empty)
WORK=${WORK:-$HOME/work}   # working dir on the build box (placeholder; was a hardcoded absolute path)
gpu_free() { u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1); [ "$u" -lt 3000 ]; }
wait_gpu() { c=0; while [ $c -lt 2 ]; do if gpu_free && ! pgrep -f "fb27.py|probe2_box.py|run_fb4.sh|qf.sh" >/dev/null; then c=$((c+1)); else c=0; fi; sleep 20; done; }
export PATH=$HOME/v2env/bin:/usr/local/cuda-13.2/bin:$PATH CUDA_HOME=/usr/local/cuda-13.2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd ~/work; mkdir -p fbout7
step() { out=$1; shift; for t in 1 2 3; do [ -f $out ] && return; wait_gpu; "$@"; done; }
step out2/ex_base.json python probe2_box.py $WORK/tb2_hf ex_base fbs7r/exam_probe.jsonl > fbout7/ex_base.log 2>&1
step out2/ex_f165k.json python probe2_box.py $WORK/loaded_k16_f165k ex_f165k fbs7r/exam_probe.jsonl > fbout7/ex_f165k.log 2>&1
FBDIR=$WORK/fbs7r SHUF=Alarm,Weather,Music,Calendar,Lists,News,Transport NEWLAB=Recipe NEIGH=1200 NS=12,25 step fbout7/fb_Recipe_f165k.json python fb27s.py $WORK/loaded_k16_f165k fbout7/fb_Recipe_f165k.json 62,63 1.0 8 1.0 3 > fbout7/fb_Recipe_f165k.log 2>&1
echo G_DONE > fbout7/G_DONE
