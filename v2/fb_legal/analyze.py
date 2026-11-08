import json,sys,numpy as np,collections,math
def wil(k,n,z=1.96):
    if n==0: return 0
    p=k/n;d=1+z*z/n;c=p+z*z/(2*n);h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n));return (c-h)/d
def an(path):
    R=json.load(open(path)); runs=R["runs"]; base_t=R["base"]["jb_teach"]; meta=R["neigh_meta"]
    A=lambda s:np.array(list(map(int,s)))
    cells=collections.defaultdict(list)
    for i,(src,g,_) in enumerate(meta): cells[(src,g)].append(i)
    out=[]
    for m in sorted({r["m"] for r in runs}):
        for n in sorted({r["n"] for r in runs if r["m"]==m}):
            sel=lambda c:[r for r in runs if r["m"]==m and r["n"]==n and r["c"]==c]
            re=sel("real")
            if not re: continue
            tm=lambda c,k="teach":np.mean([A(r[k]).mean() for r in sel(c)])
            nb=np.stack([A(r["neigh"]) for r in re]); ret=nb.mean(); tot=nb.size
            cm=min(nb[:,ix].mean() for ix in cells.values() if len(ix)>=30)
            # worst draw retention
            gains=[A(r["teach"]).mean()-base_t for r in re]; sh=[A(r["teach"]).mean() for r in sel("shuffle")]
            out.append(dict(m=m,n=n,draws=len(re),base_teach=round(base_t,3),real=round(tm("real"),3),shuf=round(tm("shuffle"),3),wrong=round(tm("wrong"),3),mixed=round(tm("mixed_wrong"),3),
                ood=round(tm("real","ood"),3),neigh=round(ret,4),neigh_wil=round(wil(int(nb.sum()),tot),4),worst_draw=round(nb.mean(1).min(),4),min_cell=round(cm,3),
                real_minus_shuf=round(tm("real")-tm("shuffle"),3)))
    return R,out
if __name__=="__main__":
    for p in sys.argv[1:]:
        R,o=an(p); print("==",p,"base",R["base"],"natural errors",R["base_labels"], "valid",R.get("validity"),"quantbase",R.get("quant_base"))
        print("%-10s %3s %2s base real shuf wrong mixed ood | neigh wilson worst mincell"%("method","n","d"))
        for x in o: print("%-10s %3d %2d %.2f %.3f %.3f %.3f %.3f %.3f | %.4f %.4f %.4f %.3f"%(x["m"],x["n"],x["draws"],x["base_teach"],x["real"],x["shuf"],x["wrong"],x["mixed"],x["ood"],x["neigh"],x["neigh_wil"],x["worst_draw"],x["min_cell"]))
