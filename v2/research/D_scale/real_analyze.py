"""python real_analyze.py BASE.json CAND.json...  retention overall + per cell (source/role x kind) with counts, Wilson lo"""
import json,sys,math,collections
def wil(k,n,z=1.96):
    if n==0: return 0.0
    p=k/n;d=1+z*z/n;c=p+z*z/(2*n);h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n));return (c-h)/d
B=json.load(open(sys.argv[1])); Bm={r["id"]:r for r in B["rows"]}
print(f"base acc {sum(r['correct'] for r in B['rows'])/len(B['rows']):.3f} n={len(B['rows'])}")
for f in sys.argv[2:]:
    C={r["id"]:r for r in json.load(open(f))["rows"]}; g=collections.defaultdict(lambda:[0,0]); a=[0,0]
    for i,b in Bm.items():
        if b["correct"]:
            a[1]+=1; a[0]+=C[i]["correct"]; k="/".join(b["cell"]); g[k][1]+=1; g[k][0]+=C[i]["correct"]
    ok=a[0]/a[1]>=.98 and all(v[0]/v[1]>=.95 for v in g.values())
    print(f"\n{f}: retention {a[0]/a[1]*100:.2f}% ({a[0]}/{a[1]}, Wilson lo {wil(*a)*100:.1f}) cand acc {sum(r['correct'] for r in C.values())/len(C):.3f} OBSERVED-STRICT-PASS={ok}")
    for k,v in sorted(g.items()): print(f"   {k:22s} {v[0]}/{v[1]} = {v[0]/v[1]*100:.1f}% (lo {wil(*v)*100:.1f})")
