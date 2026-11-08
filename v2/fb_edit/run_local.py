"""Run edit.py's run() on a plain GPU box (Vast) without Modal: stubs the modal module. usage: python run_local.py cfg.json [name]"""
import sys, json, types, os
class _Any:
    def __init__(s,*a,**k): pass
    def __getattr__(s,n): return _Any()
    def __call__(s,*a,**k): return _Any()
m = types.ModuleType("modal")
class App:
    def __init__(s,*a,**k): pass
    def function(s,*a,**k): return lambda f: f
m.App = App; m.Image = _Any(); m.Volume = _Any(); sys.modules["modal"] = m
os.makedirs("/r", exist_ok=True)
if not os.path.exists("/root/data_tox"): os.symlink(os.path.abspath(os.path.dirname(__file__) + "/../fb_legal/data_tox"), "/root/data_tox")
if not os.path.exists("/root/data_news"): os.symlink(os.path.abspath(os.path.dirname(__file__) + "/data_news"), "/root/data_news")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import edit
cfgs = json.load(open(sys.argv[1]))
for c in cfgs:
    if len(sys.argv) > 2 and c["name"] != sys.argv[2]: continue
    edit.run(c)
