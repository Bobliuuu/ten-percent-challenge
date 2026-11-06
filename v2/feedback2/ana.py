"""feedback2 analysis (local numpy on frozen features). No weight training, no retrieval of stored items.
usage: python ana.py <featdir> <tag> -> prints tables, writes results_<tag>.json"""
import numpy as np, json, sys
D=sys.argv[1]; TAG=sys.argv[2]; LI=int(sys.argv[3]) if len(sys.argv)>3 else 2   # layer idx: 0 mid,1 3/4,2 final
def ld(n):
    z=np.load(f"{D}/{TAG}_{n}.npz"); return {k:z[k] for k in z.files}
def feat(z): return z["H"][:,LI].astype(np.float64)
def lsm(L):
    L=L-L.max(1,keepdims=True); return L-np.log(np.exp(L).sum(1,keepdims=True))
class Head:
    """closed-form Gaussian-prototype head. State = projection P (r x d), class means (n_c x r) + counts, bias. No stored items."""
    def __init__(s,Xc,yc,r=96,alpha=0.1):
        s.mu0=Xc.mean(0); Z=Xc-s.mu0
        U,S,Vt=np.linalg.svd(Z,full_matrices=False); s.P=Vt[:r]          # PCA (shipped, from calib)
        Zp=Z@s.P.T; W=np.zeros_like(Zp)
        for c in np.unique(yc): W[yc==c]=Zp[yc==c]-Zp[yc==c].mean(0)
        C=W.T@W/len(W); s.Ci=np.linalg.inv((1-alpha)*C+alpha*np.trace(C)/r*np.eye(r)); s.r=r
    def proj(s,X): return (X-s.mu0)@s.P.T
def lda_scores(head,mus,x,logpri=0.0):
    Zx=head.proj(x); M=np.array(mus)@head.Ci
    return Zx@M.T-0.5*np.einsum("cr,cr->c",M,np.array(mus))[None,:]+logpri
def run_new_label(tag,tasks=("ag_news","dbpedia","emotion"),ks=(1,5,10,25),seeds=8,lam_grid=(0.0,0.3,0.6,1.0,2.0)):
    out=[]
    HOLD={"ag_news":3,"dbpedia":12,"emotion":2}
    for tn in tasks:
        h=HOLD[tn]; cal=ld(f"hold_{tn}_calib"); pool=ld(f"hold_{tn}_pool"); dev=ld(f"hold_{tn}_dev")
        full_cal=ld(f"mdc_calib"); 
        nold=int(cal["y"].max())          # index of Other = nold ; old classes 0..nold-1 (in hold-space)
        # old-class prototypes from calib items that are NOT the new class (new class never seen by the shipped system)
        keep=cal["ytrue"]!=h; Xc=feat(cal)[keep]; yc=cal["y"][keep]
        head=Head(Xc,yc)
        mus_old=[head.proj(Xc[yc==c]).mean(0) for c in range(nold)]
        n_old=np.array([ (yc==c).sum() for c in range(nold)])
        Xd=feat(dev); Ld=lsm(dev["L"][:, :nold+1]); yt=dev["ytrue"]; yd=dev["y"]    # yd==nold means new
        Xp=feat(pool); Lp=lsm(pool["L"][:, :nold+1])
        # base prediction in hold space: letter argmax; correct if == yd (Other counts as correct for new-label items => base "recall" of new label = Other-rate; we report it)
        def predict(mus_new,x,L,lam,bias_new,pure=False):
            S=lda_scores(head,list(mus_old)+[mus_new],x)               # old classes + new
            Lb=np.concatenate([L[:,:nold],L[:,nold:nold+1]*0+L[:,nold:nold+1]+bias_new,L[:,nold:nold+1]],1)  # base score for new = logp(Other)+bias
            Lb=np.concatenate([L[:,:nold],L[:,nold:nold+1]+bias_new],1)
            return (-1e9*np.ones_like(S) if pure else Lb)*(0 if pure else 1)+lam*S if not pure else S
        newlab=nold   # head-space idx of new class
        res_by={}
        for k in ks:
            for lam in lam_grid+(None,):
                rr=[]
                for sd in range(seeds):
                    rs=np.random.RandomState(sd); pidx=np.where(pool["ytrue"]==h)[0]; sel=rs.permutation(pidx)[:k]
                    # Bayesian prototype for the new class: prior mean = global mean of old protos (weight n0), data = k annotated items
                    n0=2.0; mu_new=(n0*np.mean(mus_old,0)+head.proj(Xp[sel]).sum(0))/(n0+k)
                    if lam is None: pr=predict(mu_new,Xd,Ld,1.0,0.0,pure=True)
                    else: pr=predict(mu_new,Xd,Ld,lam,0.0)
                    pred=pr.argmax(1)   # new idx = nold ; same index as Other in base
                    yy=yd.copy()
                    is_new=yt==h; old=~is_new
                    rr.append(dict(new_acc=float((pred[is_new]==nold).mean()),old_acc=float((pred[old]==yy[old]).mean()),
                                   fa=float((pred[old]==nold).mean()),
                                   base_new=float((Ld[is_new].argmax(1)==nold).mean()),base_old=float((Ld[old].argmax(1)==yy[old]).mean()),
                                   all_acc=float((np.where(pred==nold,nold,pred)==np.where(is_new,nold,yy)).mean())))
                m={q:float(np.mean([x[q] for x in rr])) for q in rr[0]}
                out.append(dict(task=tn,k=k,lam=lam,**m))
    return out
if __name__=="__main__":
    r=run_new_label(TAG)
    for x in r: print({k:(round(v,3) if isinstance(v,float) else v) for k,v in x.items()})
