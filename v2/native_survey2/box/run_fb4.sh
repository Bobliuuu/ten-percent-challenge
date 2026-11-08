export PATH=$HOME/v2env/bin:/usr/local/cuda-13.2/bin:$PATH CUDA_HOME=/usr/local/cuda-13.2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd ~/work; mkdir -p fbout
while ! grep -aq F165_DONE qf.log; do sleep 20; done
NEIGH=1200 NS=5,12,25 python fb27.py /home/ubuntu/work/tb2_hf fbout/full_62_63_m8.json 62,63 1.0 8 1.0 8 > fbout/full_62_63_m8.log 2>&1
echo FB_FULL_DONE
