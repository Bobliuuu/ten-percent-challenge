export PATH=$HOME/v2env/bin:$PATH
cd ~/work
export KEEP=keep_p100k_n165k.json
DEV=cpu python unfold_box2.py 16 1 /home/ubuntu/work/v_k16_f165k > unf_f165.log 2>&1
python art_build.py /home/ubuntu/work/art_k16_f165k 16 1 > artf.log 2>&1
cp hadamard_min.json art_k16_f165k/hadamard.json; cp art_load.py art_k16_f165k/
python art_load.py art_k16_f165k /home/ubuntu/work/loaded_k16_f165k cpu > loadf.log 2>&1
python mkprobe_dir.py loaded_k16_f165k v_k16_f165k > mkf.log 2>&1
echo FREQ_BUILT
