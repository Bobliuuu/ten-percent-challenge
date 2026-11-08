# final-candidate (f165k, 9.934% of fp16) runner
WORK=${WORK:-$HOME/work}   # working dir on the build box (placeholder; was a hardcoded absolute path)
gpu_free() { u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1); [ "$u" -lt 3000 ]; }
wait_gpu() { c=0; while [ $c -lt 2 ]; do if gpu_free && ! pgrep -f "fb27.py|probe2_box.py|run_fb4.sh|qf.sh" >/dev/null; then c=$((c+1)); else c=0; fi; sleep 20; done; }
export PATH=$HOME/v2env/bin:/usr/local/cuda-13.2/bin:$PATH CUDA_HOME=/usr/local/cuda-13.2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd ~/work; mkdir -p fbout7
step() { out=$1; shift; for t in 1 2 3 4 5 6; do [ -f $out ] && return; wait_gpu; "$@"; done; }
step out2/s7_f165k.json python probe2_box.py $WORK/loaded_k16_f165k s7_f165k fbs7/skill7_probe.jsonl > fbout7/p_f165k.log 2>&1
NEWLAB=Jukebox NEIGH=1200 NS=12,25 step fbout7/fb_Jukebox_f165k.json python fb27s.py $WORK/loaded_k16_f165k fbout7/fb_Jukebox_f165k.json 62,63 1.0 8 1.0 3 > fbout7/fb_Jukebox_f165k.log 2>&1
NEWLAB=Briefing NEIGH=1200 NS=25 step fbout7/fb_Briefing_f165k.json python fb27s.py $WORK/loaded_k16_f165k fbout7/fb_Briefing_f165k.json 62,63 1.0 8 1.0 2 > fbout7/fb_Briefing_f165k.log 2>&1
echo F_DONE > fbout7/F_DONE
