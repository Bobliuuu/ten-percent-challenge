"""pull.py MODEL: assembles results/MODEL_r1.json from volume dense/ct_MODEL_*.json (downloads via modal volume get)"""
import sys, subprocess, json, os, glob
m = sys.argv[1]; os.makedirs("results/raw", exist_ok=True)
ls = subprocess.run(["modal", "volume", "ls", "tenpct-tasksearch", "dense"], capture_output=True, text=True).stdout.split()
out = {}
for f in ls:
    if f.startswith(f"dense/ct_{m}_"):
        n = f.split(f"ct_{m}_")[1][:-5]; p = f"results/raw/ct_{m}_{n}.json"
        if not os.path.exists(p) or n.startswith("r") and False: subprocess.run(["modal", "volume", "get", "tenpct-tasksearch", f, "results/raw/", "--force"], check=True, capture_output=True)
        out[n] = json.load(open(p))
json.dump(out, open(f"results/{m}_r1.json", "w")); print(m, list(out))
