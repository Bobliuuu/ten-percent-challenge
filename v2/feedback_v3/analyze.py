"""Aggregate results/*.json -> table (uses harness/scorecard.py paired_boot / retention). python analyze.py"""
import json,glob,sys,os,numpy as np
sys.path.insert(0,os.path.expanduser("~/ten-percent/v2/harness"))
from scorecard import paired_boot,retention
PAR={"Qwen3-0.6B":0.6,"Qwen3-1.7B":1.7,"Qwen3-4B-Instruct-2507":4.0,"Qwen3-8B":8.2}
rows=[];out=[]
for f in sorted(glob.glob("results/*.json")):
    R=json.load(open(f)); m=os.path.basename(f)[:-5].rsplit("_",1)[0] if not f.endswith("_late.json") else os.path.basename(f)[:-5]; t=R["task"]
    if not R.get("final"): continue
    bt=np.array(R.get("base_teach_arr",[0]*R["sizes"]["teach"]),float)
    for meth in ["memit_down","memit_o","lmhead_mean","lmhead_lda"]:
        F=[x for x in R["final"] if x["method"]==meth]
        if not F: continue
        gains=[];shuf=[];nb=[];dr=[]
        for x in F:
            r=np.array(x["real"]["teach"],float); s=np.array(x["shuffled"]["teach"],float)
            gains.append(r-bt); shuf.append(r-s); nb+=x["real"]["neigh"]; dr.append(x["real"]["drift"]["token_agree"])
        g=paired_boot(np.concatenate(gains)); sh=paired_boot(np.concatenate(shuf))
        nbr=np.mean(nb)
        ok=g[1]>0 and g[0]>=.15 and nbr>=.98 and sh[1]>0 and sh[0]>=.10
        rows.append(dict(model=m,task=t,method=meth,cfg=F[0]["cfg"],gain=g[0],gain_lo=g[1],neigh=nbr,real_minus_shuf=sh[0],shuf_lo=sh[1],drift_tokagree=float(np.mean(dr)),base_teach=float(bt.mean()),G2=bool(ok),done=bool(R.get("done"))))
        if R.get("w4") and meth=="memit_down" and R["w4"]["runs"]:
            qb=np.array(R["w4"]["base_teach"],float)
            for q in ["none","orig","new"]:
                W=[x for x in R["w4"]["runs"] if x["q"]==q]
                if W: rows.append(dict(model=m,task=t,method=f"W4+memit_down[{q}]",gain=float(np.mean([np.mean(np.array(x["teach"])-qb) for x in W])),neigh=float(np.mean([np.mean(x["neigh"]) for x in W])),G2=None))
json.dump(rows,open("table.json","w"),indent=1)
print("| model | task | method | teach gain (95% lo) | neighbors | real-shuf (lo) | drift tok-agree | G2 |\n|---|---|---|---|---|---|---|---|")
for r in rows:
    if r["G2"] is None: print(f"| {r['model']} | {r['task']} | {r['method']} | {r['gain']:+.2f} | {r['neigh']:.3f} | | | n/a |")
    else: print(f"| {r['model']} | {r['task']} | {r['method']} | {r['gain']:+.2f} ({r['gain_lo']:+.2f}) | {r['neigh']:.3f} | {r['real_minus_shuf']:+.2f} ({r['shuf_lo']:+.2f}) | {r['drift_tokagree']:.3f} | {'PASS' if r['G2'] else 'FAIL'} |")

# ---- size ladder fit: teach-gain / neighbor retention vs log10(params), extrapolate to 14B / 32B
PAR.update({"Qwen3-14B":14.8,"Qwen3-32B":32.8})
import collections
by=collections.defaultdict(dict)
for f in sorted(glob.glob("results/*.json"),key=lambda p:p.endswith("_late.json")):
    R=json.load(open(f)); m=R["model"].split("/")[-1]; t=R["task"]
    if not R.get("final") or m not in PAR: continue
    bt=np.array(R.get("base_teach_arr",[0]*R["sizes"]["teach"]),float)
    for meth in ["memit_down","memit_o","lmhead_mean","lmhead_lda"]:
        F=[x for x in R["final"] if x["method"]==meth]
        if F: by[(t,meth)][m]=(np.concatenate([np.array(x["real"]["teach"],float)-bt for x in F]),np.array([v for x in F for v in x["real"]["neigh"]],float))
g=np.random.default_rng(0)
print("\n## size ladder (teach gain, neighbor retention) linear in log10(params B), extrapolated\n| task | method | sizes | gain per size | slope/decade (95%) | pred gain @14B (95%) | pred gain @32B (95%) | neigh per size |\n|---|---|---|---|---|---|---|---|")
for (t,meth),d in sorted(by.items()):
    ms=sorted(d,key=lambda k:PAR[k]); x=np.log10([PAR[k] for k in ms])
    if len(ms)<2: continue
    sl=[];p14=[];p32=[]
    for _ in range(400):
        y=np.array([g.choice(d[k][0],len(d[k][0])).mean() for k in ms]); b,a=np.polyfit(x,y,1); sl.append(b);p14.append(a+b*np.log10(14.8));p32.append(a+b*np.log10(32.8))
    q=lambda v:f"{np.mean(v):+.2f} ({np.quantile(v,.025):+.2f},{np.quantile(v,.975):+.2f})"
    print(f"| {t} | {meth} | {','.join(ms)} | {','.join(f'{d[k][0].mean():+.2f}' for k in ms)} | {q(sl)} | {q(p14)} | {q(p32)} | {','.join(f'{d[k][1].mean():.3f}' for k in ms)} |")
