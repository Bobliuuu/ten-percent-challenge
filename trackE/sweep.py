import modal, json, sys
app=modal.App("tenpct-trackE")
T="/Users/jhome/ten-percent/"
img=(modal.Image.debian_slim().pip_install("torch","transformers","datasets","numpy")
  .add_local_file(T+"splits.json","/root/splits.json").add_local_file(T+"trackG/dev2.py","/root/trackG/dev2.py")
  .add_local_file(T+"trackG/dev2.json","/root/trackG/dev2.json").add_local_file(T+"trackG/dev2_hard.json","/root/trackG/dev2_hard.json")
  .add_local_file(T+"trackE/lib.py","/root/lib.py").add_local_file(T+"trackE/compress.py","/root/compress.py").add_local_file(T+"trackE/vocab_ids.json","/root/vocab_ids.json"))
@app.function(image=img,gpu="L4",cpu=8,memory=32768,timeout=3000,max_containers=40)
def run(plan):
    import sys; sys.path.insert(0,"/root")
    import numpy as np
    import compress as C
    sd=C.load_base(); d=C.data(); D=C.Net(sd)
    SETS=dict(d2=d["d2"],hard=d["h2"],olddev=d["dev"])
    B={k:C.predict(D,v) for k,v in SETS.items()}
    P=C.build(sd,plan,C.batches(d["calib"],64)); r={}
    for k,v in SETS.items(): r[k]=float((C.predict(P,v).argmax(1)==B[k].argmax(1)).mean())
    r["d2acc"]=float((C.predict(P,SETS["d2"]).argmax(1)==np.array(d["l2"])).mean())
    r["params"]=P.params(); r["ratio"]=P.params()/C.BASE_PARAMS
    return dict(plan=plan,**r)
@app.local_entrypoint()
def main(out="sweep1.json"):
    plans=[mk(nl,nh,nf,128) for nl in (12,8,6,4,3,2) for nh in (12,6,3) for nf in (3072,1536,768,384)]
    plans+= [mk(12,12,3072,None),mk(12,12,3072,None,vocab=False)]
    res=list(run.map(plans,return_exceptions=True))
    res=[r for r in res if isinstance(r,dict)]
    json.dump(res,open(out,"w"))
    for r in sorted(res,key=lambda r:r["ratio"]): print(f'{r["plan"]["tag"]:18s} {r["ratio"]:.3f} d2acc {r["d2acc"]:.3f} agree d2 {r["d2"]:.4f} hard {r["hard"]:.4f} old {r["olddev"]:.4f}')
