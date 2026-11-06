import sys; sys.argv=["x","feats"]
import numpy as np, ana_fb as a, itertools
orig=a.ld
def mk(cols):
    def ld(n):
        z=orig(n); 
        if "X" in z: z["X"]=z["X"][:,cols]
        return z
    return ld
for name,cols in (("both",slice(0,5120)),("last",slice(0,2560)),("mean",slice(2560,5120))):
    a.ld=mk(cols)
    for R,AL in ((96,0.1),(48,0.1),(96,0.5),(192,0.3)):
        a.R=R; a.ALPHA=AL; res=[]
        for tn in ("dbpedia","ag_news"):
            P=dict(n0=8,w=20,lam=0.3); r,o=a.scenA(tn,P,"pool")
            res.append((tn,round(o[10]["gated"]["new"],2),round(o[10]["gated"]["old"],2),round(r["base"]["old"],2)))
        print(name,R,AL,res,flush=True)
