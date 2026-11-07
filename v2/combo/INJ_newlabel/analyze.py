import json,sys,glob,numpy as np,collections,os
B=1000; rng=np.random.default_rng(0)
arr=lambda s:np.frombuffer(s.encode(),dtype=np.uint8)-48
def wilson(k,n,z=1.96):
    p=k/n; d=1+z*z/n; c=p+z*z/(2*n); r=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n)); return (c-r)/d
def boot(M,grp):
    """M: draws x items (float diff). cluster bootstrap over draws and item groups. returns mean, lo, hi"""
    ug,inv=np.unique(grp,return_inverse=True); G=len(ug); cols=[np.where(inv==g)[0] for g in range(G)]
    out=[]
    for _ in range(B):
        di=rng.integers(0,M.shape[0],M.shape[0]); gi=rng.integers(0,G,G); ci=np.concatenate([cols[g] for g in gi])
        out.append(M[np.ix_(di,ci)].mean())
    return float(M.mean()),float(np.percentile(out,2.5)),float(np.percentile(out,97.5))
CONT=["wrong","mixed_wrong","mixed_own","shuffle"]
def cell(R,m,n):
    runs=[r for r in R["runs"] if r["m"]==m and r["n"]==n]
    by=collections.defaultdict(dict)
    for r in runs: by[r["c"]][r["d"]]=r
    ds=sorted(by["real"]); T=lambda c:np.stack([arr(by[c][d]["teach"]) for d in ds if d in by[c]]).astype(float)
    O=lambda c:np.stack([arr(by[c][d]["ood"]) for d in ds if d in by[c]]).astype(float)
    tg=np.array(R["teach_grp"]); og=np.array(R["ood_grp"]); b=arr(R["base_vec"]["jb_teach"]).astype(float); bo=arr(R["base_vec"]["jb_ood"]).astype(float)
    real=T("real"); out=dict(model=R["model"].split("/")[-1],m=m,n=n,draws=len(ds),teach_real=float(real.mean()),teach_base=float(b.mean()),nsites=float(np.mean([by["real"][d]["nsites"] for d in ds])))
    out["gain"]=boot(real-b[None],tg)
    for c in CONT:
        if c in by and len(by[c])==len(ds): out["r-"+c]=boot(real-T(c),tg); out["teach_"+c]=float(T(c).mean())
    nb=np.stack([arr(by["real"][d]["neigh"]) for d in ds]).astype(float); N=nb.shape[1]
    out["neigh"]=float(nb.mean()); out["neigh_wlo"]=float(wilson(nb.mean()*N,N)); out["neigh_min_draw"]=float(nb.mean(1).min()); out["neigh_N"]=N
    meta=R["neigh_meta"]; srcs=collections.defaultdict(list)
    for j,(i,s,g,gr) in enumerate(meta): srcs[f"{s}/{g}"].append(j)
    out["neigh_by"]={k:float(nb[:,v].mean()) for k,v in srcs.items()}
    out["ood_gain"]=boot(O("real")-bo[None],og)
    for c in ("wrong","shuffle"):
        if c in by and len(by[c])==len(ds): out["ood_r-"+c]=boot(O("real")-O(c),og)
    if all("neigh" in by["wrong"][d] for d in ds): out["neigh_wrong"]=float(np.mean([arr(by['wrong'][d]['neigh']).mean() for d in ds]))
    if all("neigh" in by["shuffle"][d] for d in ds if d in by["shuffle"]) and len(by["shuffle"])==len(ds): out["neigh_shuffle"]=float(np.mean([arr(by['shuffle'][d]['neigh']).mean() for d in ds]))
    gate_ctl=all(out.get("r-"+c,(0,-1,0))[1]>0 and out.get("r-"+c,(0,-1,0))[0]>=0.10 for c in CONT)
    gate_ctl_strict=all(out.get("r-"+c,(0,-1,0))[1]>0 and out.get("r-"+c,(0,-1,0))[0]>=0.10 for c in ["wrong","shuffle","mixed_wrong"])
    out["G2_gain"]=out["gain"][0]>=0.15 and out["gain"][1]>0; out["G2_ctl_all"]=gate_ctl; out["G2_ctl_noOwn"]=gate_ctl_strict; out["G2_neigh"]=out["neigh"]>=0.98
    out["G2"]=bool(out["G2_gain"] and gate_ctl and out["G2_neigh"]); out["G2_noOwn"]=bool(out["G2_gain"] and gate_ctl_strict and out["G2_neigh"])
    return out
if __name__=="__main__":
    rows=[]
    for f in sorted(glob.glob(sys.argv[1]+"/Qwen3-*.json")):
        R=json.load(open(f)); ms=sorted({r["m"] for r in R["runs"]})
        for m in ms:
            for n in sorted({r["n"] for r in R["runs"] if r["m"]==m}):
                try: rows.append(cell(R,m,n))
                except Exception as e: print("skip",f,m,n,repr(e)[:80])
    json.dump(rows,open(sys.argv[1]+"/table.json","w"),indent=1)
    f=lambda x:"%+.2f(%+.2f)"%(x[0],x[1]); p=lambda x:"%+.2f"%x[0]
    print("| model | method | n | draws | teach real | gain (lo) | r-wrong (lo) | r-mixwrong (lo) | r-mixown (lo) | r-shuffle (lo) | neigh | W-lo | min-draw | OOD gain (lo) | G2 | G2 w/o mixed_own |\n|"+"---|"*16)
    for o in rows:
        g=lambda k:f(o[k]) if k in o else "-"
        print(f"| {o['model']} | {o['m']} | {o['n']} | {o['draws']} | {o['teach_real']:.2f} | {g('gain')} | {g('r-wrong')} | {g('r-mixed_wrong')} | {g('r-mixed_own')} | {g('r-shuffle')} | {o['neigh']:.3f} | {o['neigh_wlo']:.3f} | {o['neigh_min_draw']:.3f} | {g('ood_gain')} | {'PASS' if o['G2'] else 'fail'} | {'PASS' if o['G2_noOwn'] else 'fail'} |")
