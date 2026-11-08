#!/bin/bash
cd ~/ten-percent/v2/fb_ablation; mkdir -p results/modal
for J in A1 A1b A1c F2 G G4 A2 B1 B2 C D E F X_SKILL7_bf16 X_SKILL7_q4 X_BANK6_bf16 X_BANK6_q4; do modal volume get --force tenpct-fbabl /$J/res.json results/modal/$J.json >/dev/null 2>&1; done
python3 - <<'PY'
import json,glob
for p in sorted(glob.glob("results/modal/*.json")):
    try: R=json.load(open(p)); print(p, "done" if R.get("done") else "partial", len(R["runs"]), "runs")
    except Exception as e: print(p,"unreadable")
PY
python3 - <<'PY'
import json, os, shutil
M = {"X1": "X_SKILL7_bf16", "X2": "X_SKILL7_q4", "X3": "X_BANK6_bf16", "X4": "X_BANK6_q4", "G": "G"}
for k, v in M.items():
    s = f"results/vast_{k}/out/res.json"; d = f"results/modal/{v}.json"
    if os.path.exists(s):
        try:
            a = json.load(open(s)); b = json.load(open(d)) if os.path.exists(d) else None
            if b is None or (not b.get("done") and (a.get("done") or len(a["runs"]) > len(b["runs"]))): shutil.copy(s, d); print("synced", k, "->", v, len(a["runs"]), "done" if a.get("done") else "partial")
        except Exception as e: print("sync err", k, e)
PY
