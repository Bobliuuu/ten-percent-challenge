"""Run push.py's run() on a plain GPU box (no Modal). usage: python run_local.py cfgs.json out.jsonl [idx ...]
Stubs the modal module; expects /root/base/{dev,calib,pool}.jsonl and writable /hf."""
import sys, types, json, os
class _Any:
    def __getattr__(s,k): return _Any()
    def __call__(s,*a,**k):
        if len(a)==1 and callable(a[0]) and not k: return a[0]
        return _Any()
m=types.ModuleType("modal")
class App:
    def __init__(s,*a,**k): pass
    def function(s,*a,**k): return lambda f:f
    def local_entrypoint(s,*a,**k): return lambda f:f
class Image:
    @staticmethod
    def debian_slim(*a,**k): return _Any()
class Volume:
    @staticmethod
    def from_name(*a,**k):
        v=_Any(); v.commit=lambda:None; return v
m.App=App; m.Image=Image; m.Volume=Volume
sys.modules["modal"]=m
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import push
cfgs=json.load(open(sys.argv[1])); out=sys.argv[2]
sel=[int(a) for a in sys.argv[3:]] or range(len(cfgs))
for i in sel:
    try:
        r=push.run(cfgs[i]); r["_idx"]=i
    except Exception as e:
        import traceback; traceback.print_exc(); r={"_idx":i,"error":repr(e),"cfg":cfgs[i]}
    open(out,"a").write(json.dumps(r,default=str)+"\n")
