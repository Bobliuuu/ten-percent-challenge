#!/bin/bash
# make_bundle.sh JOB CFG.json  -> bundles/JOB (code+data+cfg+run.sh); launch with research/V_vast/launch_vast.py JOB --bundle ...
set -e; cd "$(dirname "$0")"; B=bundles/$1; rm -rf $B; mkdir -p $B/results; cp fblib.py run_abl.py $B/; cp -r data $B/data; cp $2 $B/cfg.json
cat > $B/run.sh <<'EOS'
#!/bin/bash
cd /workspace/bundle; mkdir -p results; pip install -q transformers==4.51.3 accelerate sentencepiece 2>&1 | tail -1
HF_HOME=/workspace/hf python run_abl.py cfg.json results/out > results/run.log 2>&1; touch results/DONE
EOS
echo bundle $B
