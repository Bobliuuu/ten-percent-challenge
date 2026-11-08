import json,sys,subprocess,os,numpy as np,collections,math
import analyze2
m,pre=sys.argv[1],sys.argv[2]
ls=subprocess.run(["modal","volume","ls","tenpct-tasksearch","dense"],capture_output=True,text=True).stdout.split()
R={}
os.makedirs("results/raw",exist_ok=True)
for f in ls:
    if f.startswith(f"dense/ct_{m}_{pre}_"):
        n=f.split(f"ct_{m}_")[1][:-5]; p=f"results/raw/ct_{m}_{n}.json"
        subprocess.run(["modal","volume","get","tenpct-tasksearch",f,"results/raw/","--force"],capture_output=True)
        R[n]=json.load(open(p))
json.dump(R,open(f"results/{m}_{pre}.json","w"))
base={r["id"]:r for r in R[pre+"_bf16"]["rows"]}
cf=[c for c in R if not c.endswith("bf16")]; cf.sort(key=lambda c:R[c]["ratio"])
print(m,pre,{c:(round(R[c]["ratio"]*100,2),round(R[c]["body_bpw"],2)) for c in cf})
kinds=sorted(set(r["kind"] for r in base.values()))
bc=[i for i,r in base.items() if r["correct"]]
print("base acc",len(bc)/len(base), "median margin", np.median([base[i]["m_min"] for i in bc]))
for c in cf:
    cr={r["id"]:r for r in R[c]["rows"]}; k=sum(cr[i]["correct"] for i in bc); lo,_=analyze2.wil(k,len(bc))
    per={kd:(sum(cr[i]["correct"] for i in bc if base[i]["kind"]==kd)/max(1,sum(1 for i in bc if base[i]["kind"]==kd)),sum(1 for i in bc if base[i]["kind"]==kd)) for kd in kinds}
    print(c,f"ret {k/len(bc):.3f} lo {lo:.3f}", {kd:(round(v[0],3),v[1]) for kd,v in per.items()})
print("-- per source x class")
for c in cf:
    cr={r["id"]:r for r in R[c]["rows"]}; d=collections.defaultdict(lambda:[0,0])
    for i in bc:
        k=(base[i]["src"],base[i]["kind"]); d[k][1]+=1; d[k][0]+=cr[i]["correct"]
    print(c,{f"{k[0][:6]}/{k[1][:5]}":f"{v[0]/v[1]:.2f}({v[1]})" for k,v in sorted(d.items())})
bs=collections.defaultdict(list)
for i,r in base.items(): bs[r["src"]].append(r["correct"])
print("base acc by src",{k:round(sum(v)/len(v),3) for k,v in bs.items()})
