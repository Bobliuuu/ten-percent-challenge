import time
from compress import *
sd=load_base(); d=data(); D=Net(sd)
SETS=dict(d2=d["d2"],hard=d["h2"],olddev=d["dev"])
B={k:predict(D,v) for k,v in SETS.items()}
print("base acc d2",(B["d2"].argmax(1)==np.array(d["l2"])).mean(),"hard",(B["hard"].argmax(1)==np.array(d["hl"])).mean(),flush=True)
cb=batches(d["calib"],64)
def run(name,plan):
    t=time.time(); P=build(sd,plan,cb); r=[]
    for k,v in SETS.items():
        p=predict(P,v).argmax(1); r.append(f"{k}:agree {(p==B[k].argmax(1)).mean():.4f}")
    p=predict(P,SETS["d2"]).argmax(1)
    print(f"{name}: params {P.params()/BASE_PARAMS:.4f} d2acc {(p==np.array(d['l2'])).mean():.4f} "+" ".join(r)+f" ({time.time()-t:.0f}s)",flush=True)
if __name__=="__main__":
    full=[(i,12,3072) for i in range(12)]
    run("identity",dict(layers=full))
    run("vocab",dict(layers=full,vocab=True))
    run("vocab+rank128",dict(layers=full,vocab=True,rank=128))
    run("ffn1024",dict(layers=[(i,12,1024) for i in range(12)],vocab=True))
    run("6 layers",dict(layers=[(i,12,3072) for i in (0,2,4,6,8,11)],vocab=True))
