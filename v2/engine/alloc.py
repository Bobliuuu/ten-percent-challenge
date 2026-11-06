"""Merge TABLE_JSON shards (from modal logs / out_tables.json / volume) and allocate per-module levels L for a target body bpw.
usage: python3 alloc.py tables.json target_bpw [target ...]  -> writes levels_<target>.json
Lagrangian: minimize sum rel_err(L) + lam*bits(L). bits = ent*n + 16/g*n (scales)."""
import json,sys,glob
def load(paths):
    T={};g=128
    for p in paths:
        d=json.load(open(p)); d=d if isinstance(d,dict) else d
        for k,v in d["table"].items(): T[k]=v
        g=d["g"]
    return T,g
def alloc(T,g,target):
    mods=list(T); N=sum(T[m]["n"] for m in mods)
    opts={m:[(int(L),e,T[m]["n"]*(b+16/g)) for L,(e,b) in T[m]["tab"].items()] for m in mods}
    def pick(lam):
        ch={};bits=0
        for m in mods:
            L,e,b=min(opts[m],key=lambda o:o[1]+lam*o[2]); ch[m]=L; bits+=b
        return ch,bits
    lo,hi=1e-14,1e3
    for _ in range(200):
        lam=(lo*hi)**.5; ch,b=pick(lam)
        if b/N>target: lo=lam
        else: hi=lam
    ch,b=pick(hi); return ch,b/N
if __name__=="__main__":
    T,g=load(glob.glob("table_*.json")) if False else load(sys.argv[1].split(","))
    for t in map(float,sys.argv[2:]):
        ch,b=alloc(T,g,t); json.dump(ch,open(f"levels_{t}.json","w")); 
        from collections import Counter; print(t,round(b,3),Counter(ch.values()))
