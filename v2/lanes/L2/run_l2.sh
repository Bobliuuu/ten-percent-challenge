export PATH=$HOME/v2env/bin:/usr/local/cuda-13.2/bin:$PATH CUDA_HOME=/usr/local/cuda-13.2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd ~/work; mkdir -p fbs7 out2 fbout7
while ! grep -aq FB_FULL_DONE run_fb4.log 2>/dev/null && pgrep -f run_fb4.sh >/dev/null; do sleep 20; done
# 1) SKILL7 7-label base vs candidate (vLLM greedy, real generated word)
python probe2_box.py /home/ubuntu/work/tb2_hf s7_base fbs7/skill7_probe.jsonl > fbout7/p_base.log 2>&1
python probe2_box.py /home/ubuntu/work/v_k16_n184k s7_cand fbs7/skill7_probe.jsonl > fbout7/p_cand.log 2>&1
# 2) feedback with truly unknown labels on the candidate (unfolded), real/shuffled/empty
for lab in Jukebox Briefing; do NEWLAB=$lab NEIGH=1200 NS=12,25 python fb27s.py /home/ubuntu/work/loaded_k16_n184k fbout7/fb_${lab}_cand.json 62,63 1.0 8 1.0 3 > fbout7/fb_${lab}_cand.log 2>&1; done
NEWLAB=Jukebox NS=12 python fb27s.py /home/ubuntu/work/tb2_hf fbout7/fb_Jukebox_base.json 62,63 1.0 8 1.0 3 > fbout7/fb_Jukebox_base.log 2>&1
echo L2_DONE > fbout7/L2_DONE
