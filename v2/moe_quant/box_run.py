"""run cfgs on a plain GPU box (Vast): python box_run.py cfgs.json name1 name2 ...  (layout: /root/code = this dir + data/splits dirs; /r = results+hf cache)"""
import sys, json, os
sys.path.insert(0, "/root/code")
os.makedirs("/r/res", exist_ok=True); os.environ["HF_HOME"] = "/r/hf"
class V:
    def commit(s): pass
    def reload(s): pass
import core
cfgs = json.load(open(sys.argv[1])); names = sys.argv[2:]
core.main([c for n in names for c in cfgs if c["name"] == n], V())
