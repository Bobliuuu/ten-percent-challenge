"""usage: python run_fb.py cfg.json [name] -- runs edit_s7.run() on a plain GPU box (modal stubbed). Outputs /r/<name>.json"""
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
os.makedirs(os.environ.get("EDIT_OUT", "/r"), exist_ok=True); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import edit_s7
for c in json.load(open(sys.argv[1])):
    if len(sys.argv) > 2 and c["name"] != sys.argv[2]: continue
    edit_s7.run(c)
