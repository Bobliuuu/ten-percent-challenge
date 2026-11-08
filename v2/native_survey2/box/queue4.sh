export PATH=$HOME/v2env/bin:/usr/local/cuda-13.2/bin:$PATH CUDA_HOME=/usr/local/cuda-13.2
cd ~/work
while pgrep -f "[q]ueue3.sh" > /dev/null; do sleep 10; done
while ! grep -aq "^DONE" unf_v4.log; do sleep 10; done
for pair in "/home/ubuntu/work/v_k16_n140k v_k16_n140k"; do set -- $pair; [ -f out/$2.json ] || python probe_box.py $1 $2 > probe_$2.log 2>&1; done
while ! grep -aq "^DONE" unf_v5.log; do sleep 10; done
for pair in "/home/ubuntu/work/v_k8_n170k v_k8_n170k"; do set -- $pair; [ -f out/$2.json ] || python probe_box.py $1 $2 > probe_$2.log 2>&1; done
