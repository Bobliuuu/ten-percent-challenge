"""Run dense._run on a local GPU box (no Modal). usage: DENSE_HF=... DENSE_DATA=... python local_run.py cfgs.json [index ...]"""
import sys, types, json, os
class _A:
    def __init__(s,*a,**k): pass
    def __getattr__(s,n): return lambda *a,**k: (lambda f:f) if n in("function","local_entrypoint") else _A()
    def __call__(s,*a,**k): return _A()
m=types.ModuleType("modal"); m.App=lambda *a,**k:_A(); m.Image=_A(); m.Volume=types.SimpleNamespace(from_name=lambda *a,**k:_A()); sys.modules["modal"]=m
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import dense
cs=json.load(open(sys.argv[1])); idx=[int(i) for i in sys.argv[2:]] or range(len(cs))
for i in idx: 
    c=cs[i]; tag=c['model'].split('/')[-1]; hd=os.environ.get('DENSE_HF','/hf')
    fn=f"{hd}/dense/base_{tag}.json" if c.get('mode')=='base' else (f"{hd}/dense/t1_{tag}_{c.get('name')}.json" if (c.get('t1') or c.get('mode')=='t1base') else None) or f"{hd}/dense/res_{tag}_{c.get('name','x')}.json"
    if os.path.exists(fn) and 'multi' not in c: print('SKIP',fn,flush=True); continue
    print("JOB",i,cs[i].get("name"),flush=True); dense._run(cs[i])
