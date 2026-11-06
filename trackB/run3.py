import numpy as np, sys
from transformers import AutoTokenizer
from lex import LexMemory
C=np.load("cache.npz",allow_pickle=True)
tok=AutoTokenizer.from_pretrained("textattack/bert-base-uncased-SST-2")
ids=lambda T:[x[1:-1] for x in tok(list(T),truncation=True,max_length=64)["input_ids"]]
dt,ft,st=ids(C["dev_t"]),ids(C["feedback_t"]),ids(C["syn_t"])
dl,fl,sl,dy,fy,sy,sw=C["dev_l"],C["feedback_l"],C["syn_l"],C["dev_y"],C["feedback_y"],C["syn_y"],C["syn_w"]
acc=lambda l,y:float((l.argmax(1)==y).mean())
diff=lambda L:L[:,1]-L[:,0]
def go(seed=1,n_train=8,n_anchor=300,use_err=True,**kw):
    rng=np.random.RandomState(seed); words=list(dict.fromkeys(sw)); rng.shuffle(words)
    batches=[words[i:i+6] for i in range(0,24,6)]
    err=np.where(fl.argmax(1)!=fy)[0]; err_tr,err_te=err[:12],err[12:]
    ok=np.where(fl.argmax(1)==fy)[0]; anc=rng.choice(ok,n_anchor,replace=False)
    M=LexMemory(**kw)
    for i in anc: M.add_anchor(ft[i],fy[i],diff(fl[i:i+1])[0])
    tr=[];te=[]
    for b in batches:
        t1=[];t2=[]
        for w in b:
            ix=rng.permutation(np.where(sw==w)[0]); t1+=list(ix[:n_train]); t2+=list(ix[n_train:])
        tr.append(t1); te.append(np.array(t2))
    P=lambda L,T,idx:acc(M.predict(L[idx],[T[i] for i in idx]),(sy if T is st else fy if T is ft else dy)[idx])
    def ev(name):
        r=dict(step=name,dev=acc(M.predict(dl,dt),dy),errH=P(fl,ft,err_te))
        for b in range(4): r[f"b{b}"]=P(sl,st,te[b])
        r["bytes"]=M.bytes(); r["nw"]=len(M.w); return r
    rows=[ev("empty")]
    for b in range(4):
        for i in tr[b]: M.add(st[i],sy[i],diff(sl[i:i+1])[0])
        if b==0 and use_err:
            for i in err_tr: M.add(ft[i],fy[i],diff(fl[i:i+1])[0])
        M.fit_pending(); rows.append(ev(f"batch{b}"))
    return rows
if __name__=="__main__":
    for kw in [dict(),dict(lr=0.25,margin=4.0),dict(margin=10.0),dict(epochs=60)]:
        rs=[go(seed=s,**kw) for s in range(3)]
        f=lambda k:np.mean([r[-1][k] for r in rs])
        print(kw,"dev %.3f errH %.3f syn %s bytes %d"%(f("dev"),f("errH"),[round(float(f(f"b{b}")),3) for b in range(4)],f("bytes")),flush=True)
