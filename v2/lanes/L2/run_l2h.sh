# ship the diffuse edit as a real low-rank feedback delta (feedback.pt = U,V factors of the down_proj edit) and verify it reproduces the in-memory result
gpu_free() { u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1); [ "$u" -lt 3000 ]; }
wait_gpu() { c=0; while [ $c -lt 2 ]; do if gpu_free && ! pgrep -f "fb27.py|probe2_box.py|run_fb4.sh|qf.sh|fb27s.py" >/dev/null; then c=$((c+1)); else c=0; fi; sleep 20; done; }
export PATH=$HOME/v2env/bin:/usr/local/cuda-13.2/bin:$PATH CUDA_HOME=/usr/local/cuda-13.2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd ~/work
while [ ! -f fbout7/G_DONE ]; do sleep 15; done
wait_gpu
FBDIR=/home/ubuntu/work/fbs7r SHUF=Alarm,Weather,Music,Calendar,Lists,News,Transport NEWLAB=Recipe NEIGH=1200 NS=25 SAVE_FB=fbout7/feedback_Recipe.pt python fb27s.py /home/ubuntu/work/loaded_k16_f165k fbout7/fb_Recipe_save.json 62,63 1.0 8 1.0 1 > fbout7/fb_Recipe_save.log 2>&1
echo H_DONE > fbout7/H_DONE
