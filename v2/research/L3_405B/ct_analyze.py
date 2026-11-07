"""usage: python ct_analyze.py BASE_ct.json CAND_ct.json ... ; per task retention (overall, per role S1/S3, per kind), Wilson lo, strict gates (>=98 overall, >=95 every cell)"""
import json,sys,math,collections
def wil(k,n,z=1.96):
    if n==0: return 0.0
    p=k/n;d=1+z*z/n;c=p+z*z/(2*n);h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n));return (c-h)/d
B={r["id"]:r for r in json.load(open(sys.argv[1]))["rows"]}
print("base acc per task:",{t:round(sum(r["correct"] for r in B.values() if r["task"]==t)/sum(1 for r in B.values() if r["task"]==t),3) for t in sorted({r["task"] for r in B.values()})})
for f in sys.argv[2:]:
    C={r["id"]:r for r in json.load(open(f))["rows"]}; print("\n==",f)
    for t in sorted({r["task"] for r in B.values()}):
        g=collections.defaultdict(lambda:[0,0])
        for i,b in B.items():
            if b["task"]==t and b["correct"]:
                for key in ("ALL","role:"+b["role"],"kind:"+b["kind"]): g[key][1]+=1; g[key][0]+=C[i]["correct"]
        cells=[v for k,v in g.items() if k!="ALL"]
        allv=g["ALL"]; ret=allv[0]/allv[1]; worst=min(v[0]/v[1] for v in cells if v[1]>=10)
        ok=ret>=.98 and wil(*allv)>=.98 and worst>=.95
        print(f"{t:8s} ret {ret:.3f} (lo {wil(*allv):.3f}; {allv[0]}/{allv[1]}) worst cell {worst:.3f} PASS={ok} | "+"; ".join(f"{k} {v[0]/v[1]:.2f}({v[1]})" for k,v in sorted(g.items()) if k!="ALL"))
