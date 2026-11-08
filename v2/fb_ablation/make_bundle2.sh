#!/bin/bash
# make_bundle2.sh JOB CFG SCRIPT -> bundles/JOB with fblib.py, run_abl.py, run_exam.py, data, exam_data
set -e; cd "$(dirname "$0")"; B=bundles/$1; rm -rf $B; mkdir -p $B/results; cp fblib.py run_abl.py run_exam.py $B/; cp -r data exam_data $B/; cp $2 $B/cfg.json
cat > $B/run.sh <<EOS
#!/bin/bash
cd /workspace/bundle; mkdir -p results; pip install -q transformers==4.51.3 accelerate sentencepiece 2>&1 | tail -1
HF_HOME=/workspace/hf python $3 cfg.json results/out > results/run.log 2>&1; touch results/DONE
EOS
echo bundle $B
