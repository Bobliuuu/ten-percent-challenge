import sys; from core import *
lo,hi=int(sys.argv[1]),int(sys.argv[2]); out=sys.argv[3]
D=data(); m=fullbase().to(DEV); 
Hs=collect_H(m,D["B_dev"]+D["B_cal"][:0]) if False else None
import pickle
Hf=HERE+"/H_fp.pt"
if os.path.exists(Hf): Hs=torch.load(Hf)
else: raise SystemExit("run mkH first")
bl=logits(m,D["B_cal"])
res={}
t0=time.time()
BITS=[2,3,4]; GS=[64,128,256]
for l in range(lo,hi):
    for n in LIN:
        mod=m.bert.encoder.layer[l].get_submodule(n); W0=mod.weight.data.clone(); key=f"L{l}.{n}"
        for b in BITS:
            for g in GS:
                q,s,z=gptq2(W0.cpu(),Hs[key],b,g)
                mod.weight.data=dequant(q,s,z,g).to(DEV)
                kl,ag=kl_agree(logits(m,D["B_cal"]),bl)
                res[f"{key}|{b}|{g}"]=dict(kl=kl,ag=ag,size=est_size(q,z,b))
        mod.weight.data=W0
    print("layer",l,f"{time.time()-t0:.0f}s",flush=True); json.dump(res,open(out,"w"))
