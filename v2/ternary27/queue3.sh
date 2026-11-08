export PATH=$HOME/v2env/bin:/usr/local/cuda-13.2/bin:$PATH CUDA_HOME=/usr/local/cuda-13.2
cd ~/work
while pgrep -f "[q]ueue2.sh" > /dev/null || ! grep -aq "^DONE" unf_v3.log; do sleep 10; done
for pair in "/home/ubuntu/work/v_k8_full v_k8_full" "prism-ml/Bonsai-27B-unpacked bonsai27_1bit" "Qwen/Qwen3.6-27B q36_27b"; do set -- $pair; [ -f out/$2.json ] || python probe_box.py $1 $2 > probe_$2.log 2>&1; done
