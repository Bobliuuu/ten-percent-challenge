import json,glob,sys,os,numpy as np
sys.path.insert(0,os.path.expanduser("~/ten-percent/v2/harness"))
from scorecard import retention
out={}; md=""
for model in ["Qwen3-14B","Qwen3-8B"]:
    bf=f"results/t1/t1_{model}_bf16.json"
    if not os.path.exists(bf): continue
    base=json.load(open(bf))["t1"]; bc=np.array([r["correct"] for r in base]); kind=np.array([r["kind"] for r in base]); src=np.array([r["source"] for r in base])
    md+=f"\n### {model}  (bf16 base item accuracy: overall {bc.mean():.3f}; " + ", ".join(f"{s} {bc[src==s].mean():.3f}" for s in ['S1','S3'])+")\n\n| config | bytes | ratio | ret overall (Wilson lo) | S1 | S3 | nocall | single | multi | worst kind | cand acc |\n|---|---|---|---|---|---|---|---|---|---|---|\n"
    rows=[]
    for f in sorted(glob.glob(f"results/t1/t1_{model}_*.json")):
        n=os.path.basename(f)[len(f"t1_{model}_"):-5]
        if n=="bf16": continue
        d=json.load(open(f)); cc=np.array([r["correct"] for r in d["t1"]])
        o=retention(bc,cc); ks={k:retention(bc,cc,kind==k) for k in ["nocall","single","multi"]}; ss={s:retention(bc,cc,src==s) for s in ["S1","S3"]}
        worst=min(v["ret"] for v in ks.values())
        rows.append(dict(cfg=n,bytes=d["bytes"],ratio=d["ratio"],ret=o["ret"],ret_lo=o["ret_lo"],worst=worst,kinds={k:v["ret"] for k,v in ks.items()},src={k:v["ret"] for k,v in ss.items()},acc=float(cc.mean())))
    rows.sort(key=lambda r:r["ratio"])
    for r in rows:
        md+=f"| {r['cfg']} | {int(r['bytes']):,} | {r['ratio']:.4f} | {r['ret']:.3f} ({r['ret_lo']:.3f}) | {r['src']['S1']:.3f} | {r['src']['S3']:.3f} | {r['kinds']['nocall']:.3f} | {r['kinds']['single']:.3f} | {r['kinds']['multi']:.3f} | {r['worst']:.3f} | {r['acc']:.3f} |\n"
    fam={"generic recipe (generic calib, RTN-proxy alloc)":[r for r in rows if r["cfg"] in("t096","t11","t125","t15","t30")],
         "task-calib + KL-sensitivity alloc (R3)":[r for r in rows if r["cfg"].startswith("cal_kl_r") or r["cfg"]=="t096_cal_kl"]}
    for fn,fr in fam.items():
        if len(fr)>=3:
            x=np.log([r["ratio"] for r in fr])
            for key,get,tgt in [("overall",lambda r:r["ret"],0.98),("worst_kind",lambda r:r["worst"],0.95)]:
                v=np.array([get(r) for r in fr]); y=np.log(np.clip(1-v,1e-3,1)/np.clip(v,1e-3,1)); A=np.polyfit(x,y,1); xt=(np.log((1-tgt)/tgt)-A[1])/A[0]
                md+=f"\nFamily [{fn}] {key}: logit(fail)~log(ratio) slope {A[0]:.2f}; ret>={tgt} at ratio ~{np.exp(xt):.3f} ({len(fr)} pts)\n"
    # extrapolation: log-odds of failure (1-ret) vs log ratio, linear fit
    if len(rows)>=3:
        x=np.log([r["ratio"] for r in rows]); 
        for key,get,tgt in [("overall",lambda r:r["ret"],0.98),("worst_kind",lambda r:r["worst"],0.95)]:
            y=np.log(np.clip(1-np.array([get(r) for r in rows]),1e-3,1)/np.clip(np.array([get(r) for r in rows]),1e-3,1))
            A=np.polyfit(x,y,1); yt=np.log((1-tgt)/tgt); xt=(yt-A[1])/A[0]
            md+=f"\nFit logit(1-ret)~log(ratio) [{key}]: slope {A[0]:.2f}; ret>={tgt} predicted at ratio ~{np.exp(xt):.3f} (extrapolation from {len(rows)} points; CI not meaningful with this few points)\n"
    out[model]=rows
json.dump(out,open("t1_retention.json","w"),indent=1,default=float); open("t1_retention.md","w").write(md); print(md)
