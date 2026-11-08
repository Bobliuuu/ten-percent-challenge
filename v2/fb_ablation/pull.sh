#!/bin/bash
cd ~/ten-percent/v2/fb_ablation; mkdir -p results/modal
for J in A1 A1b A2 B1 B2 C D E F X_SKILL7_bf16 X_SKILL7_q4 X_BANK6_bf16 X_BANK6_q4; do modal volume get --force tenpct-fbabl /$J/res.json results/modal/$J.json >/dev/null 2>&1; done
python3 - <<'PY'
import json,glob
for p in sorted(glob.glob("results/modal/*.json")):
    try: R=json.load(open(p)); print(p, "done" if R.get("done") else "partial", len(R["runs"]), "runs")
    except Exception as e: print(p,"unreadable")
PY
