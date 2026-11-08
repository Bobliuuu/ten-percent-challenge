import os, sys, json
os.environ.update(DENSE_HF="/root/out", DENSE_DATA="/root/ct/dense_data.json", CT_ITEMS="/root/ct/ct_items.jsonl", CT_CALIB="/root/ct/calib_mix.jsonl", DENSE_GPU="1")
os.makedirs("/root/out/dense", exist_ok=True); sys.path.insert(0, "/root/ct")
import dense_core
M = "Qwen/Qwen2.5-32B-Instruct"
cfgs = [dict(model=M, name="b32_bf16", mode="ctbase", ct_bs=2),
        dict(model=M, name="s32_r0985", ctasks=1, calib_ct=1, target_ratio=0.0985, emb_L=4, head_L=4, g=256, ct_bs=2, ncal=128)]
res = dense_core._run(dict(multi=cfgs)); print("DONE", [r for r in res if isinstance(r, dict) and "error" in r])
