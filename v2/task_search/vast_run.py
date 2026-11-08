"""Vast/anywhere runner for TASK_SEARCH2: python vast_run.py jobs_vast.json <idx...>  (same job schema as modal_run.py; results -> /root/out/ct_*.json)"""
import json, sys, os
os.environ.update(HF_HOME="/root/hf", DENSE_HF="/root/out", DENSE_DATA="/root/code/dense_data.json", CT_ITEMS="/root/code/t2_items.jsonl", CT_CALIB="/root/code/t2_calib.jsonl", HF_HUB_ENABLE_HF_TRANSFER="0")
os.makedirs("/root/out/dense", exist_ok=True); sys.path.insert(0, "/root/code")
import dense_core
jobs = json.load(open(sys.argv[1]))
for k in sys.argv[2:]:
    job = jobs[int(k)]
    if job.get("gpu_load", True): os.environ["DENSE_GPU"] = "1"
    else: os.environ.pop("DENSE_GPU", None)
    cfgs = [dict(model=job["model"], ctasks=1, calib_ct=1, gen=True, gen_bs=(12 if "32B" in job["model"] else 32), ct_bs=(2 if "32B" in job["model"] else 8), **c) if not c["name"].endswith("bf16") else dict(model=job["model"], name=c["name"], mode="ctbase", gen=True, gen_bs=(12 if "32B" in job["model"] else 32), ct_bs=(2 if "32B" in job["model"] else 8)) for c in job["cfgs"]]
    print("RUN", job["model"], [c["name"] for c in cfgs], flush=True)
    res = dense_core._run(dict(multi=cfgs)); print("DONE", job["model"], [r for r in res if isinstance(r, dict) and "error" in r], flush=True)
