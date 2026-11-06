import sys,json,time
import numpy as np
import compress as C
from compress import mk
sd=C.load_base(); d=C.data(); D=C.Net(sd)
SETS=dict(d2=d["d2"],hard=d["h2"],olddev=d["dev"])
B={k:C.predict(D,v) for k,v in SETS.items()}
cb=C.batches(d["calib"],64)
cfgs=json.loads(sys.argv[1]); out=sys.argv[2]
res=[]
for c in cfgs:
    t=time.time(); plan=mk(*c); P=C.build(sd,plan,cb); r={}
    for k,v in SETS.items(): r[k]=float((C.predict(P,v).argmax(1)==B[k].argmax(1)).mean())
    r["d2acc"]=float((C.predict(P,SETS["d2"]).argmax(1)==np.array(d["l2"])).mean()); r["params"]=P.params(); r["ratio"]=P.params()/C.BASE_PARAMS
    r["plan"]=plan; res.append(r); json.dump(res,open(out,"w"))
    print(f'{plan["tag"]:18s} {r["ratio"]:.3f} d2acc {r["d2acc"]:.3f} agree d2 {r["d2"]:.4f} hard {r["hard"]:.4f} old {r["olddev"]:.4f} ({time.time()-t:.0f}s)',flush=True)
