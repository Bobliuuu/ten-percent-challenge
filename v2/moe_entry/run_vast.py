"""Run core.main on a Vast box. layout: /root/code/{core.py,fb.py,run_vast.py}, /root/data (voice_*.jsonl), /root/splits (news), /r/res (results). usage: python run_vast.py cfgs.json"""
import sys, json, os
os.environ.setdefault("HF_HOME", "/r/hf"); os.makedirs("/r/res", exist_ok=True); sys.path.insert(0, "/root/code")
import core
class V:
    def commit(self): pass
core.main(json.load(open(sys.argv[1])), V())
