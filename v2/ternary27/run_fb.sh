export PATH=$HOME/v2env/bin:/usr/local/cuda-13.2/bin:$PATH CUDA_HOME=/usr/local/cuda-13.2
cd ~/work; mkdir -p fbout
while ! grep -aq PROBES_DONE run_probes.log; do sleep 20; done
# sweep (n=12, 2 draws): layers x m
for cfg in "63 1.0 8 1.0" "62,63 1.0 8 1.0" "60,61,62,63 1.0 8 1.0" "56,58,60,62 1.0 8 1.0" "62,63 1.0 16 1.0"; do set -- $cfg; NS=12 python fb27.py /home/ubuntu/work/tb2_hf fbout/sweep_${1}_m${3}.json $1 $2 $3 $4 2 > fbout/sweep_${1}_m${3}.log 2>&1; done
echo FB_SWEEP_DONE
