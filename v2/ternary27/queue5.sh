export PATH=$HOME/v2env/bin:/usr/local/cuda-13.2/bin:$PATH CUDA_HOME=/usr/local/cuda-13.2
cd ~/work
while pgrep -f "[q]ueue4.sh" > /dev/null; do sleep 10; done
while ! grep -aq "MKPROBE" mk3.log; do sleep 10; done
python probe_box.py /home/ubuntu/work/loaded_k16_n140k loaded_k16_n140k > probe_loaded_k16.log 2>&1
