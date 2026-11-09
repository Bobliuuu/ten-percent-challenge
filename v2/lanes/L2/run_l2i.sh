# closing gaps: (1) parent Qwen3.8-27B on the shared exam, (2) shipped CLIs end to end, (3) bytes audit on the exact tree
WORK=${WORK:-$HOME/work}   # working dir on the build box (placeholder; was a hardcoded absolute path)
gpu_free() { u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1); [ "$u" -lt 3000 ]; }
wait_gpu() { c=0; while [ $c -lt 2 ]; do if gpu_free && ! pgrep -f "fb27.py|probe2_box.py|fb27s.py|run_fb4.sh|qf.sh" >/dev/null; then c=$((c+1)); else c=0; fi; sleep 20; done; }
export PATH=$HOME/v2env/bin:/usr/local/cuda-13.2/bin:$PATH CUDA_HOME=/usr/local/cuda-13.2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HOME=${HF_HOME:-$WORK/hf}
cd ~/work; mkdir -p fbout7 L2out
wait_gpu; [ -f out2/ex_parent.json ] || python probe2_box.py Qwen/Qwen3.8-27B ex_parent fbs7r/exam_probe.jsonl > fbout7/ex_parent.log 2>&1
echo PARENT_DONE > L2out/PARENT_DONE
S=~/work/ship_L2; rm -rf $S; mkdir -p $S; cp -al art_k16_f165k/. $S/; cp L2/answer.py L2/feedback.py L2/fb27s.py L2/art_load.py $S/; mkdir -p $S/data/exam; cp L2/data/exam/cal.jsonl L2/data/exam/neigh.jsonl $S/data/exam/
export L2_UNFOLD=~/work/unf_ship; rm -rf $L2_UNFOLD
wait_gpu; python $S/answer.py $S "play some jazz music please" > L2out/cli_smoke.txt 2>&1
wait_gpu; python $S/feedback.py $S L2/data/ann25.jsonl Recipe --unfolded $L2_UNFOLD > L2out/fb_real.log 2>&1; mv $S/feedback.pt L2out/fb_real.pt
wait_gpu; python $S/feedback.py $S L2/data/ann25.jsonl Recipe --unfolded $L2_UNFOLD --shuffle-labels > L2out/fb_shuf.log 2>&1; mv $S/feedback.pt L2out/fb_shuf.pt
for arm in empty real shuf; do rm -f $S/feedback.pt; [ $arm != empty ] && cp L2out/fb_$arm.pt $S/feedback.pt; wait_gpu; python $S/answer.py $S --batch L2/data/l2_neighbour_eval_rows.jsonl --out L2out/cli_$arm.json > L2out/cli_$arm.log 2>&1; done
rm -f $S/feedback.pt; cp L2out/fb_real.pt $S/feedback.pt
python ~/work/L2/bytes_audit.py $S --base-fp16-bytes 53808408928 --json L2out/bytes_audit.json > L2out/bytes_audit.txt 2>&1
echo I_DONE > L2out/I_DONE
