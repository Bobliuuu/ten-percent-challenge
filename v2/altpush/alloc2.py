"""Allocate per-module ECSQ step c (as int c*100) from table_*.json shards (push.py mode=table).
usage: python3 alloc2.py out_prefix sbits g target_bpw[,..] shard1.json shard2.json ...
bits/weight = entropy + sbits/g. Lagrangian min sum w_m*relerr_m + lam*bits, w_m = 1 (optionally sens weights). Writes levels_<target>.json {"oli.name": c*100}."""
import json,sys
pre,sbits,g=sys.argv[1],int(sys.argv[2]),int(sys.argv[3]); targets=[float(x) for x in sys.argv[4].split(",")]
T={}
for p in sys.argv[5:]:
    d=json.load(open(p)); T.update(d["table"])
def tab(m): return [(int(L),e,T[m]["n"]*(b+sbits/g)) for L,(e,b) in T[m]["tab"].items()]
N=sum(T[m]["n"] for m in T)
for tg in targets:
    lo,hi=1e-14,1e3
    def pick(lam):
        ch={};bits=0
        for m in T:
            L,e,b=min(tab(m),key=lambda o:o[1]+lam*o[2]); ch[m]=L; bits+=b
        return ch,bits
    for _ in range(200):
        lam=(lo*hi)**.5; ch,b=pick(lam)
        if b/N>tg: lo=lam
        else: hi=lam
    ch,b=pick(hi); json.dump(ch,open(f"{pre}_levels_{tg}.json","w")); 
    from collections import Counter; print(tg,round(b/N,3),sorted(Counter(ch.values()).items()))
