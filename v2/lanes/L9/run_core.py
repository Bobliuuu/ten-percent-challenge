"""usage: python run_core.py CFGS.json OUT   (original in-memory fake-quant path on the HF base; data in l9/data, base outputs seeded from results/)"""
import sys, json, os, shutil
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import core235
cf, OUT = sys.argv[1:3]; os.makedirs(OUT + "/res", exist_ok=True)
for sn in ("SEL", "REP"):
    if not os.path.exists(f"{OUT}/res/base_skill7_{sn}_1.json"): shutil.copy(f"{HERE}/results/outs_skill7_base_{sn}.json", f"{OUT}/res/base_skill7_{sn}_1.json")
core235.main(json.load(open(cf)), None, OUT, HERE + "/data")
