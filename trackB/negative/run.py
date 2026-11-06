import numpy as np, json, sys
from memory import Memory
C=np.load("cache.npz",allow_pickle=True)
FS=slice(0,768)
de,dl,dy=C["dev_e"][:,FS],C["dev_l"],C["dev_y"]; fe,fl,fy=C["feedback_e"][:,FS],C["feedback_l"],C["feedback_y"]
se,sl,sy,sw=C["syn_e"][:,FS],C["syn_l"],C["syn_y"],C["syn_w"]
acc=lambda l,y:float((l.argmax(1)==y).mean())
def scenario(seed=1,n_train=8,**kw):
    rng=np.random.RandomState(seed)
    words=list(dict.fromkeys(sw)); rng.shuffle(words)
    batches=[words[i:i+6] for i in range(0,24,6)]   # 4 sequential feedback batches
    M=Memory(**kw); tr_idx=[];te_by_batch=[]
    for w_ in batches:
        te=[]
        for w in w_:
            ix=rng.permutation(np.where(sw==w)[0]); tr_idx.append(ix[:n_train]); te+=list(ix[n_train:])
        te_by_batch.append(np.array(te))
    # batch 0 also gets the base's own mistakes on the feedback split
    err=np.where(fl.argmax(1)!=fy)[0]; half=len(err)//2
    err_tr,err_te=err[:half],err[half:]
    rows=[]
    def log(name):
        r={"step":name,"dev":acc(M.predict(dl,de),dy),"err_heldout":acc(M.predict(fl[err_te],fe[err_te]),fy[err_te]),
           "fb_all_nonerr":acc(M.predict(fl,fe),fy)}
        for b,te in enumerate(te_by_batch): r[f"syn_b{b}"]=acc(M.predict(sl[te],se[te]),sy[te])
        r["bytes"]=M.bytes(); r["n"]=len(M.y); rows.append(r)
    log("base(empty)")
    for b,w_ in enumerate(batches):
        for w in w_:
            for i in tr_idx[[x for x in range(len(tr_idx))][batches.index(w_)*6+list(w_).index(w)]]:
                # store only when memory+base is currently wrong (compact); dedup handled by add
                if M.predict(sl[i:i+1],se[i])[0].argmax()!=sy[i]: M.add(se[i],sy[i])
        if b==0:
            for i in err_tr:
                if M.predict(fl[i:i+1],fe[i])[0].argmax()!=fy[i]: M.add(fe[i],fy[i])
        log(f"after batch {b}")
    return rows,M
if __name__=="__main__":
    base={"dev":acc(dl,dy),"syn_all":acc(sl,sy),"fb_errs":int((fl.argmax(1)!=fy).sum())}; print(base)
    for kw in [dict(),dict(tau=0.8),dict(tau=0.85,k=3),dict(tau=0.95),dict(alpha=4.0)]:
        rs=[scenario(seed=s,**kw)[0] for s in range(3)]
        print(kw)
        last=rs[0][-1]
        avg={k:np.mean([r[-1][k] for r in rs]) for k in last if k!="step"}
        print({k:round(v,4) for k,v in avg.items()})
