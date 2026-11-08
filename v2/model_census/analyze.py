"""Pull census results from volume, rank by retention (real gen). usage: python3 analyze.py"""
import json,subprocess,os,collections,math
def wil(k,n,z=1.96):
    if n==0: return 0,0
    p=k/n; d=1+z*z/n; c=(p+z*z/2/n)/d; h=z*math.sqrt(p*(1-p)/n+z*z/4/n/n)/d; return c-h,c+h
os.makedirs("raw",exist_ok=True)
ls=subprocess.run(["modal","volume","ls","tenpct-tasksearch","dense"],capture_output=True,text=True).stdout.split()
models=collections.defaultdict(dict)
for f in ls:
    n=os.path.basename(f)
    if n.startswith("ct_") and (n.endswith("_bf16.json") or n.endswith("_c099.json")):
        p="raw/"+n
        if not os.path.exists(p): subprocess.run(["modal","volume","get","tenpct-tasksearch",f,"raw/","--force"],capture_output=True)
        if os.path.exists(p):
            d=json.load(open(p)); models[n[3:].rsplit("_",1)[0]][n.rsplit("_",1)[1][:-5]]=d
rows=[]
for m,d in models.items():
    if "bf16" not in d or "c099" not in d: continue
    b={r["id"]:r for r in d["bf16"]["rows"]}; q={r["id"]:r for r in d["c099"]["rows"]}
    if not set(q)<=set(b) or not any("gen_correct" in r for r in q.values()): continue
    bc=[i for i,r in b.items() if r.get("gen_correct")]
    if not bc: rows.append(dict(model=m,base=0)); continue
    k=sum(q[i].get("gen_correct",False) for i in bc); lo,_=wil(k,len(bc))
    cells=collections.defaultdict(lambda:[0,0])
    for i in bc: c=(b[i]["task"],b[i]["src"],b[i]["kind"]); cells[c][1]+=1; cells[c][0]+=q[i].get("gen_correct",False)
    ws=min(((v[0]/v[1]),c,v[1]) for c,v in cells.items() if v[1]>=8) if any(v[1]>=8 for v in cells.values()) else (None,None,0)
    pt=collections.defaultdict(lambda:[0,0])
    for i in bc: pt[b[i]["task"]][1]+=1; pt[b[i]["task"]][0]+=q[i].get("gen_correct",False)
    rows.append(dict(model=m,base=len(bc)/len(b),n=len(bc),ret=k/len(bc),lo=lo,ratio=d["c099"].get("ratio"),bpw=d["c099"].get("body_bpw"),worst=ws[0],worstcell="/".join(ws[1]) if ws[1] else "",task={t:round(v[0]/v[1],3) for t,v in pt.items()}))
rows.sort(key=lambda r:-(r.get("lo") or 0))
json.dump(rows,open("census_probe.json","w"),indent=1)
L=["| rank | model | base acc | n base-correct | ratio of fp16 | body bpw | retention | Wilson lo | worst cell (n>=8) | per task |","|---|---|---|---|---|---|---|---|---|---|"]
for i,r in enumerate(rows,1):
    if "ret" not in r: L.append(f"| {i} | {r['model']} | 0 | | | | | | | |"); continue
    L.append(f"| {i} | {r['model']} | {r['base']:.3f} | {r['n']} | {(r['ratio'] or 0)*100:.2f}% | {(r['bpw'] or 0):.2f} | {r['ret']*100:.1f}% | {r['lo']*100:.1f}% | {(r['worst'] or 0)*100:.0f}% {r['worstcell']} | {r['task']} |")
open("probe_table.md","w").write("\n".join(L)); print("\n".join(L))
