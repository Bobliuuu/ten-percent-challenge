export PATH=$HOME/v2env/bin:/usr/local/cuda-13.2/bin:$PATH CUDA_HOME=/usr/local/cuda-13.2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd ~/work; mkdir -p fbout; rm -f fbout/*
for cfg in "62,63 1.0 8 1.0" "60,61,62,63 1.0 8 1.0"; do set -- $cfg; NS=12 python fb27.py /home/ubuntu/work/tb2_hf fbout/sweep_${1}_m${3}.json $1 $2 $3 $4 2 > fbout/sweep_${1}_m${3}.log 2>&1; done
echo FB_SWEEP_DONE
