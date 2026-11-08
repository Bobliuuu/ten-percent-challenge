import json,sys,glob,os,math,collections
def wil(k,n,z=1.96):
    if n==0: return 0,0
    p=k/n; d=1+z*z/n; c=(p+z*z/(2*n))/d; h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d; return c-h,c+h
D=os.path.dirname(os.path.abspath(__file__))+"/results_vast"
base={(m,p):{r["id"]:r for r in json.load(open(f"{D}/base/ct_Qwen2.5-{m}-Instruct_{p}_bf16.json"))["rows"]} for p in "in" for m in ("14B","32B")}
out=[]
for f in sorted(glob.glob(f"{D}/*/dense/ct_Qwen2.5-*.json")):
    mm=f.split("ct_Qwen2.5-")[1].split("-Instruct")[0]; n=f.split("-Instruct_")[1][:-5]
    if n.endswith("bf16"): continue
    d=json.load(open(f)); p=n[0]; b=base[(mm,p)]; cr={r["id"]:r for r in d["rows"]}
    bc=[i for i in b if b[i]["correct"]]; k=sum(cr[i]["correct"] for i in bc); lo,_=wil(k,len(bc))
    cell=collections.defaultdict(lambda:[0,0]); src=collections.defaultdict(lambda:[0,0])
    for i in bc:
        c=cell[(b[i]["src"],b[i]["kind"])]; c[1]+=1; c[0]+=cr[i]["correct"]
        s=src[b[i]["role"]]; s[1]+=1; s[0]+=cr[i]["correct"]
    worst=min((v[0]/v[1],k_) for k_,v in cell.items() if v[1]>=8)
    out.append((mm+" "+n,d["ratio"]*100,d["body_bpw"],k/len(bc),lo,{r:round(v[0]/v[1],3) for r,v in sorted(src.items())},worst))
for o in sorted(out): print(f"{o[0]:18s} {o[1]:5.2f}% bpw{o[2]:.2f} ret {o[3]:.3f} lo {o[4]:.3f} bySrc {o[5]} worst {o[6][0]:.2f} {o[6][1]}")
