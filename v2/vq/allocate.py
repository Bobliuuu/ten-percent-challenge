# usage: allocate.py sens.json budget_bits_per_w name out.json [gamma]
# greedy per-matrix logk allocation from single-matrix sensitivity (kl at slogk), minimising sum kl s.t. mean bits/w <= budget
import json,sys,re,heapq
sens=json.load(open(sys.argv[1])); B=float(sys.argv[2]); name=sys.argv[3]; out=sys.argv[4]
gamma=float(sys.argv[5]) if len(sys.argv)>5 else 0.87
lo,hi=int(sys.argv[6]) if len(sys.argv)>6 else 10, int(sys.argv[7]) if len(sys.argv)>7 else 16
sl=sens["slogk"]; R=sens["res"]
names=list(R); n={k:R[k]["n"] for k in names}; kl={k:max(R[k]["kl"],1e-7) for k in names}
tot=sum(n.values())
cur={k:lo for k in names}
f=lambda k,l: kl[k]*gamma**(l-sl)
bits=sum(n[k]*cur[k]/8 for k in names)/tot
h=[(-(f(k,lo)-f(k,lo+1))/(n[k]/8),k) for k in names]; heapq.heapify(h)
while h:
    _,k=heapq.heappop(h)
    if cur[k]>=hi: continue
    nb=bits+n[k]/8/tot
    if nb>B: continue
    cur[k]+=1; bits=nb
    if cur[k]<hi: heapq.heappush(h,(-(f(k,cur[k])-f(k,cur[k]+1))/(n[k]/8),k))
plan=[[f"^{re.escape(k)}$",cur[k]] for k in names]
pred=sum(f(k,cur[k]) for k in names)
from collections import Counter
print("bits/w",bits,"pred sumKL",pred,"hist",sorted(Counter(cur.values()).items()))
json.dump([{"name":name,"logk":lo,"plan":plan}],open(out,"w"))
