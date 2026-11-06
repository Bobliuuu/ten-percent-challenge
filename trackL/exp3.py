exec(open("exp.py").read().split("def LR")[0])
from transformers import AutoTokenizer
tk=AutoTokenizer.from_pretrained("textattack/bert-base-uncased-SST-2")
ids={n:[sorted(set(x)-{101,102,0}) for x in tk(t,truncation=True,max_length=64)["input_ids"]] for n,t in [("pool",pool),("d2",t2),("hard",th)]}
# (a) closed form: least squares base_diff ~ a*q+b ; and pure bias (mean residual) ; and bias that maximizes pool agreement (grid)
A=np.stack([qd["pool"],np.ones(len(pool))],1); a,b=np.linalg.lstsq(A,bd["pool"],rcond=None)[0]; print("lstsq",a,b)
rep("(a1) lstsq scale+bias",a*qd["d2"]+b,a*qd["hard"]+b,"8B")
bs=np.linspace(-2,2,81); best=bs[np.argmax([((qd["pool"]+x>0)==(bd["pool"]>0)).mean() for x in bs])]; print("grid bias",best)
rep("(a2) bias only (grid on pool agreement)",qd["d2"]+best,qd["hard"]+best,"2B")
# robust to eval-hash: also fit bias on d2 cal half? (skip)
# (b) ridge closed form with bias, pooled / cls; low-rank via PCA of pooled diff
def ridge(feat,lam,r=None):
    X=lambda k:np.concatenate([qd[k][:,None],FA[k][feat],np.ones((len(qd[k]),1))],1)
    Xp=X("pool"); w=np.linalg.solve(Xp.T@Xp+lam*np.eye(Xp.shape[1]),Xp.T@bd["pool"]); return X,w
for lam in [10,30,100,300]:
    X,w=ridge("pooled",lam); rep(f"(b) ridge full pooled lam={lam}",X("d2")@w,X("hard")@w,"1.5KB")
# diagonal-ish / low-rank: residual r=bd-(a q+b); ridge on top-r PCA of pooled
R=bd["pool"]-(a*qd["pool"]+b); Pm=FA["pool"]["pooled"].mean(0)
U,S,Vt=np.linalg.svd(FA["pool"]["pooled"]-Pm,full_matrices=False)
for r in [4,16,64]:
    Z=lambda k:(FA[k]["pooled"]-Pm)@Vt[:r].T
    Zp=np.c_[Z("pool"),np.ones(len(pool))]; w=np.linalg.solve(Zp.T@Zp+30*np.eye(r+1),Zp.T@R)
    f=lambda k:a*qd[k]+b+np.c_[Z(k),np.ones(len(qd[k]))]@w
    rep(f"(b2) rank-{r} PCA ridge residual",f("d2"),f("hard"),f"{(r*768+r)*2}B fp16")
# (c) perceptron lexical (trackB style): learn w on pool, margin on base sign, only update on disagreement/low-margin
def percep(ep=10,margin=1.0,lr=0.25,seed=0,maxdf=10**9):
    rng=np.random.RandomState(seed); w=np.zeros(30522); z0=a*qd["pool"]+b; y=np.sign(bd["pool"]); L=ids["pool"]
    df=np.zeros(30522)
    for x in L: df[x]+=1
    for e in range(ep):
        for i in rng.permutation(len(L)):
            t=L[i]; 
            if not t: continue
            n=len(t)**.5
            if y[i]*(z0[i]+w[t].sum()/n)<margin:
                tt=[u for u in t if df[u]<=maxdf]; w[tt]+=lr*y[i]/n
    return w
def lexp(w,k): return a*qd[k]+b+np.array([w[t].sum()/max(len(t),1)**.5 for t in ids[k]])
for mg,maxdf in [(0.5,10**9),(1.0,10**9),(1.0,200),(2.0,200)]:
    w=percep(margin=mg,maxdf=maxdf); wq=np.round(w*8).clip(-127,127)/8; nz=(wq!=0).sum()
    rep(f"(c) perceptron margin={mg} maxdf={maxdf}",lexp(wq,"d2"),lexp(wq,"hard"),f"nnz {nz} bytes {3*nz+8}")
# (d) kNN memory of disagreement cases: store (pooled normalized int8, base sign); gate by similarity and local agreement
def nrm(x): x=x-Pm; return x/np.linalg.norm(x,axis=1,keepdims=True)
Pp=nrm(FA["pool"]["pooled"]); dis=np.where((qd["pool"]*a+b>0)!=(bd["pool"]>0))[0]; print("pool disagreements after a,b:",len(dis))
for nagree,tau in [(0,0.9),(0,0.8),(2000,0.8),(2000,0.9),(5000,0.9)]:
    idx=np.r_[dis,np.random.RandomState(0).choice(np.setdiff1d(np.arange(len(pool)),dis),nagree,replace=False)]
    M=Pp[idx]; my=np.sign(bd["pool"][idx])
    def kp(k):
        z=a*qd[k]+b; S=nrm(FA[k]["pooled"])@M.T; j=S.argmax(1); s=S.max(1)
        return np.where(s>tau,my[j]*max(1,0)*(abs(z)+3),z)   # override with base sign when near a stored case
    rep(f"(d) kNN n_dis={len(dis)} extra_agree={nagree} tau={tau}",kp("d2"),kp("hard"),f"bytes {len(idx)*769}")
