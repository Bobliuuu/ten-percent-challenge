"""Run core.main on a plain GPU box (Vast). usage: python vast_runner.py cfgs.json ; results in /workspace/bundle/results (symlinked from /r/res)."""
import sys, os, json
os.makedirs("/workspace/bundle/results", exist_ok=True); os.makedirs("/r", exist_ok=True)
if not os.path.exists("/r/res"): os.symlink("/workspace/bundle/results", "/r/res")
if not os.path.exists("/root/data"): os.symlink("/workspace/bundle/rdata", "/root/data")
os.environ.setdefault("HF_HOME", "/workspace/hf")
sys.path.insert(0, "/workspace/bundle"); import core
class V: 
    def commit(self): pass
core.main(json.load(open(sys.argv[1])), V())
