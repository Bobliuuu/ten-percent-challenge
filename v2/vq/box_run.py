"""usage: python box_run.py tcq_run|vq8_run cfg.json NAME  (runs one cfg by name on this box; /code holds the files, /vol/res results)"""
import sys, json, os, importlib
sys.path.insert(0, "/code/stub"); sys.path.insert(0, "/code")
os.makedirs("/vol/res", exist_ok=True)
sys.modules["modal"] = importlib.import_module("modal_stub")
m = importlib.import_module(sys.argv[1]); cfg = [c for c in json.load(open(sys.argv[2])) if c["name"] == sys.argv[3]][0]
cfg.pop("size", None); cfg.pop("maxmem_list", None); cfg.pop("maxmem", None)
cfg["vllm_util"] = 0.8
print(json.dumps(m._impl(cfg), default=str))
