# retry-until-success runner; waits for a free GPU before every step (other jobs share this GPU)
gpu_free() { u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1); [ "$u" -lt 3000 ]; }
wait_gpu() { c=0; while [ $c -lt 2 ]; do if gpu_free && ! pgrep -f "fb27.py|probe2_box.py|run_fb4.sh|qf.sh" >/dev/null; then c=$((c+1)); else c=0; fi; sleep 20; done; }
export PATH=$HOME/v2env/bin:/usr/local/cuda-13.2/bin:$PATH CUDA_HOME=/usr/local/cuda-13.2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd ~/work; mkdir -p fbs7 out2 fbout7
step() { out=$1; shift; for t in 1 2 3 4 5 6; do [ -f $out ] && return; wait_gpu; "$@"; done; }
step out2/s7_base.json python probe2_box.py /home/ubuntu/work/tb2_hf s7_base fbs7/skill7_probe.jsonl > fbout7/p_base.log 2>&1
step out2/s7_cand.json python probe2_box.py /home/ubuntu/work/loaded_k16_n184k s7_cand fbs7/skill7_probe.jsonl > fbout7/p_cand.log 2>&1
for lab in Jukebox Briefing; do NEWLAB=$lab NEIGH=1200 NS=12,25 step fbout7/fb_${lab}_cand.json python fb27s.py /home/ubuntu/work/loaded_k16_n184k fbout7/fb_${lab}_cand.json 62,63 1.0 8 1.0 3 > fbout7/fb_${lab}_cand.log 2>&1; done
NEWLAB=Jukebox NS=12 step fbout7/fb_Jukebox_base.json python fb27s.py /home/ubuntu/work/tb2_hf fbout7/fb_Jukebox_base.json 62,63 1.0 8 1.0 3 > fbout7/fb_Jukebox_base.log 2>&1
echo L2_DONE > fbout7/L2_DONE
