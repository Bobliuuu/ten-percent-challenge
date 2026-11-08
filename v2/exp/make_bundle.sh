#!/bin/sh
# usage: make_bundle.sh OUTDIR   -> bundle for research/V_vast/launch_vast.py (run.sh MODEL TAG [SETS])
set -e
O="$1"; R=$(cd "$(dirname "$0")/../.." && pwd)
rm -rf "$O"; mkdir -p "$O/v2/exp" "$O/v2/entry32b" "$O/results"
cp "$R"/v2/exp/run_retention.py "$R"/v2/exp/analyze_pair.py "$R"/v2/exp/bytes.py "$R"/v2/exp/SPLITS.json "$O/v2/exp/"
cp -r "$R"/v2/exp/splits "$O/v2/exp/"
cp "$R"/v2/entry32b/scorer.py "$R"/v2/entry32b/dev_inj.jsonl "$O/v2/entry32b/"
mkdir -p "$O/v2/harness"; cp "$R"/v2/harness/stats.py "$O/v2/harness/"
cat > "$O/run.sh" <<'EOF'
#!/bin/bash
# run.sh MODEL TAG   (two passes: pass1 default batching, pass2 reversed/eager/max_num_seqs=48 = batch-order floor)
set -x; cd "$(dirname "$0")"; mkdir -p results
M="$1"; T="$2"; SETS="${3:-SEL,REP,GEN,INJ}"; TP="${4:-1}"
pip install -q "vllm==0.11.0" "transformers==4.57.1" scipy "huggingface_hub<1.0" datasets 2>&1 | tail -2
export HF_HOME=/workspace/hf
for P in 1 2; do
  python v2/exp/run_retention.py --model "$M" --exp X01 --arm "base_$T" --seed 0 --sets "$SETS" --pass $P --tp $TP --backend vllm --util 0.88 || exit 1
  cp -r v2/exp/results/X01/. results/
done
touch results/DONE
EOF
chmod +x "$O/run.sh"
