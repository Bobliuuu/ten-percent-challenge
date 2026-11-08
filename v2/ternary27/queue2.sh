export PATH=$HOME/v2env/bin:/usr/local/cuda-13.2/bin:$PATH CUDA_HOME=/usr/local/cuda-13.2
cd ~/work
for pair in "/home/ubuntu/work/tb2_hf tb2_exact" "Qwen/Qwen3.8-27B q38_27b" "/home/ubuntu/work/v_k8_n140k v_k8_n140k"; do set -- $pair; [ -f out/$2.json ] || python probe_box.py $1 $2 > probe_$2.log 2>&1; done
