"""usage: python run_arm.py cfgs.json NAME[,NAME...]   (on a >=96GB GPU box; /root/rep holds data/, dense_repair.py; results in /root/out/dense)"""
import os, sys, json
os.environ.update(DENSE_HF="/root/out", DENSE_DATA="/root/rep/data/dense_data.json", CT_ITEMS="/root/rep/data/ct_items.jsonl", CT_CALIB="/root/rep/data/calib_mix.jsonl", DENSE_GPU="1")
os.makedirs("/root/out/dense", exist_ok=True); sys.path.insert(0, "/root/rep")
import dense_repair
M = os.environ.get("REP_MODEL", "Qwen/Qwen2.5-32B-Instruct")
want = sys.argv[2].split(",")
cfgs = [dict(model=M, **c) for c in json.load(open(sys.argv[1])) if c["name"] in want]
res = dense_repair._run(dict(multi=cfgs)); print("DONE", json.dumps(res, default=str))
