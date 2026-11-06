import numpy as np, itertools
from head import ResidualMemory
C=np.load("cache.npz",allow_pickle=True)
acc=lambda l,y:float((l.argmax(1)==y).mean())
def go(feat,seed=1,n_train=8,n_anchor=300,verbose=False,**kw):
    sl_={"pooler":slice(0,768),"cls":slice(768,1536),"mean":slice(1536,2304),"all":slice(0,2304)}[feat]
    de,fe,se=[C[k][:,sl_] for k in ["dev_e","feedback_e","syn_e"]]
    dl,fl,sl,dy,fy,sy,sw=C["dev_l"],C["feedback_l"],C["syn_l"],C["dev_y"],C["feedback_y"],C["syn_y"],C["syn_w"]
    rng=np.random.RandomState(seed); words=list(dict.fromkeys(sw)); rng.shuffle(words)
    batches=[words[i:i+6] for i in range(0,24,6)]
    err=np.where(fl.argmax(1)!=fy)[0]; err_tr,err_te=err[:12],err[12:]
    ok=np.where(fl.argmax(1)==fy)[0]; anc=rng.choice(ok,n_anchor,replace=False)
    M=ResidualMemory(de.shape[1],de.mean(0),**kw)
    for i in anc: M.add_anchor(fe[i],fy[i])
    M.Abase=np.array([fl[i][1]-fl[i][0] for i in anc])
    tr=[];te=[]
    for b in batches:
        t1=[];t2=[]
        for w in b:
            ix=rng.permutation(np.where(sw==w)[0]); t1+=list(ix[:n_train]); t2+=list(ix[n_train:])
        tr.append(t1); te.append(np.array(t2))
    def ev(name):
        r=dict(step=name,dev=acc(M.predict(dl,de),dy),errH=acc(M.predict(fl[err_te],fe[err_te]),fy[err_te]))
        for b in range(4): r[f"b{b}"]=acc(M.predict(sl[te[b]],se[te[b]]),sy[te[b]])
        r["bytes"]=M.bytes(); return r
    rows=[ev("empty")]
    for b in range(4):
        for i in tr[b]: M.add(se[i],sy[i],sl[i])
        if b==0:
            for i in err_tr: M.add(fe[i],fy[i],fl[i])
        M.fit_pending(); rows.append(ev(f"batch{b}"))
    return rows
if __name__=="__main__":
    for feat in ["pooler","cls","mean","all"]:
        for lr,ep,mg in [(0.05,30,1.0),(0.2,50,1.0)]:
            rs=[go(feat,seed=s,lr=lr,epochs=ep,margin=mg) for s in range(3)]
            f=lambda k:np.mean([r[-1][k] for r in rs])
            print(feat,lr,ep,"dev %.3f errH %.3f syn %s bytes %d"%(f("dev"),f("errH"),[round(f(f"b{b}"),3) for b in range(4)],f("bytes")),flush=True)
