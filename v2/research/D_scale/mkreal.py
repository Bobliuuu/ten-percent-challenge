import json,math,os,collections
R="results/"
def wil(k,n,z=1.96):
    if n==0: return 0.0
    p=k/n;d=1+z*z/n;c=p+z*z/(2*n);h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n));return (c-h)/d
def ret(b,c):
    B={r["id"]:r for r in json.load(open(R+b))["rows"]}; C={r["id"]:r for r in json.load(open(R+c))["rows"]}; g=collections.defaultdict(lambda:[0,0]); a=[0,0]
    for i,x in B.items():
        if x["correct"]: a[1]+=1; a[0]+=C[i]["correct"]; k="/".join(x["cell"]); g[k][1]+=1; g[k][0]+=C[i]["correct"]
    return a,g
print("| artifact | task | retention [Wilson lo] | cells (kept/base-correct) | strict |\n|---|---|---|---|---|")
for n,tag in [("V7 label-mix + self tool","V7"),("V3 label-mix + gold tool","V3"),("V9 V3 + INJ items","V9"),("V12 T1-only calib","V12"),("V11 V7 + self-distilled labels","V11")]:
    for t in ("toolsel","inj"):
        c=f"real_{tag}_{t}.json"
        if not os.path.exists(R+c): continue
        a,g=ret(f"real_base72_{t}.json",c); ok=a[0]/a[1]>=.98 and all(v[0]/v[1]>=.95 for v in g.values())
        print(f"| {n} | {t.upper()} | {a[0]/a[1]*100:.2f}% [{wil(*a)*100:.1f}] ({a[0]}/{a[1]}) | "+"; ".join(f"{k} {v[0]}/{v[1]}={v[0]/v[1]*100:.1f}" for k,v in sorted(g.items()))+f" | {'PASS' if ok else 'fail'} |")
