"""feedback2 analysis (numpy, frozen features from feat.py). No weight training; no stored items; no context edits.
python ana.py <featdir> <tag> [layer_idx 0 mid|1 3/4|2 final]  -> results_<tag>.json + tables"""
import numpy as np, json, sys, itertools
D=sys.argv[1]; TAG=sys.argv[2]; LI=int(sys.argv[3]) if len(sys.argv)>3 else 2
import os
ALPHA=float(os.environ.get('ALPHA','0.1'))
KS=(1,5,10,25); SEEDS=10
def ld(n):
    z=np.load(f"{D}/{TAG}_{n}.npz"); return {k:z[k] for k in z.files}
def _z(a): a=a.astype(np.float64); return (a-a.mean(0))/(a.std(0)+1e-6)
def X(z):
    if LI==9: return z["M"].astype(np.float64)
    if LI==8: return np.concatenate([z["H"][:,1].astype(np.float64),z["M"].astype(np.float64)],1)
    return z["H"][:,LI].astype(np.float64)
def lsm(L):
    L=L-L.max(1,keepdims=True); return L-np.log(np.exp(L).sum(1,keepdims=True))
# ---------------- closed-form prototype head (state: P r x d, Ci r x r, class means + counts)
class Head:
    def __init__(s,Xc,yc,r=96,alpha=0.1):
        alpha=ALPHA
        s.mu0=Xc.mean(0); Z=Xc-s.mu0
        s.P=np.linalg.svd(Z,full_matrices=False)[2][:r]
        Zp=Z@s.P.T; W=Zp.copy()
        for c in np.unique(yc): W[yc==c]-=Zp[yc==c].mean(0)
        C=W.T@W/len(W); s.Ci=np.linalg.inv((1-alpha)*C+alpha*np.trace(C)/r*np.eye(r)); s.r=r
    def proj(s,x): return (x-s.mu0)@s.P.T
    def score(s,x,mus): M=np.asarray(mus); return s.proj(x)@(s.Ci@M.T)-0.5*np.einsum("cr,rs,cs->c",M,s.Ci,M)[None,:]
def update(mus,cnt,Zs,ys,w):
    """Bayesian running-mean update: annotated item i adds weight w to class ys[i]'s mean."""
    mus=[m.copy() for m in mus]; cnt=list(cnt)
    for z,c in zip(Zs,ys):
        mus[c]=(cnt[c]*mus[c]+w*z)/(cnt[c]+w); cnt[c]+=w
    return mus,cnt
def bar(*a): return float(np.mean(a))
# ---------------- generic scenario runner. pred fns: (state,x,L)->scores
def evalmethods(name,Hd,calL,calX,caly,annot_pool,evals,ks,seeds,methods,sel_fn):
    pass
def agg(rs): return {q:float(np.mean([r[q] for r in rs])) for q in rs[0]}

# ============ SCENARIO A: new label (head-only addition of a class the prompt never lists) ============
HOLD={"ag_news":3,"dbpedia":12,"emotion":2}
def scenA(tn,params,split_eval):
    h=HOLD[tn]; cal=ld(f"hold_{tn}_calib"); pool=ld(f"hold_{tn}_pool"); dev=ld(f"hold_{tn}_dev")
    nold=int(cal["y"].max()); keep=cal["ytrue"]!=h; Xc=X(cal)[keep]; yc=cal["y"][keep]
    head=Head(Xc,yc,r=params.get("r",96)); mus0=[head.proj(Xc[yc==c]).mean(0) for c in range(nold)]; cnt0=[float((yc==c).sum()) for c in range(nold)]
    ev=dev if split_eval=="dev" else pool
    Xe=X(ev); Le=lsm(ev["L"][:,:nold+1]); te=ev["ytrue"]; ye=ev["y"]; Xp=X(pool)
    newm=te==h; old=~newm
    def finish(pred,base=False):
        return dict(new=float((pred[newm]==nold).mean()),old=float((pred[old]==ye[old]).mean()),fa=float((pred[old]==nold).mean()),
                    f1=0.0)
    res={"base":finish(Le.argmax(1))}
    out={}
    for k in KS:
        rows={m:[] for m in ("bias","lda_pure","lda_gate","lda_comb","steer")}
        for sd in range(SEEDS):
            rs=np.random.RandomState(sd); pidx=np.where(pool["ytrue"]==h)[0]
            if split_eval=="pool": # exclude annotated from eval
                pass
            sel=rs.permutation(pidx)[:k]; Za=head.proj(Xp[sel])
            # prior for new class: mean of old protos with weight n0 -> posterior after k annotations (weight w each)
            n0=params["n0"]; mnew=(n0*np.mean(mus0,0)+params["w"]*Za.sum(0))/(n0+params["w"]*k)
            S=head.score(Xe,mus0+[mnew]); Lb=Le.copy()
            # (1) bias only: raise Other logit just enough so all k annotated items would be labelled Other
            La=lsm(pool["L"][sel][:,:nold+1]); b=max(0,(La[:,:nold].max(1)-La[:,nold]).max())
            Lb2=Le.copy(); Lb2[:,nold]+=b; rows["bias"].append(finish(Lb2.argmax(1)))
            S=S.copy(); S[:,nold]+=params.get("nb",0.0)
            rows["lda_pure"].append(finish(S.argmax(1)))
            pg=Le.argmax(1).copy(); pg[S.argmax(1)==nold]=nold; rows["lda_gate"].append(finish(pg))
            Sc=Le+params["lam"]*S; rows["lda_comb"].append(finish(Sc.argmax(1)))
            # (4) steering at final layer: v = mean(annotated) - mean(old calib) ; gated shift on Other logit via cos
            v=Xp[sel].mean(0)-Xc.mean(0); xe=Xe-Xc.mean(0); cos=(xe@v)/(np.linalg.norm(xe,axis=1)*np.linalg.norm(v)+1e-9)
            g=1/(1+np.exp(-(cos-params["st"])/0.05)); Ls=Le.copy(); Ls[:,nold]+=params["sa"]*g; rows["steer"].append(finish(Ls.argmax(1)))
        out[k]={m:agg(r) for m,r in rows.items()}
    return res,out
def tuneA(tn):
    best=None
    for n0,w,nb in itertools.product((0.5,2),(2,6),(0,2,4,6)):
        pr=dict(n0=n0,w=w,lam=0.3,nb=nb,st=.2,sa=3,r=96); res,o=scenA(tn,pr,"pool")
        sc=bar(o[5]["lda_pure"]["new"],o[5]["lda_pure"]["old"],o[10]["lda_pure"]["new"],o[10]["lda_pure"]["old"])-3*max(0,res["base"]["old"]-o[5]["lda_pure"]["old"]-0.01)
        if best is None or sc>best[0]: best=(sc,pr)
    # steer tune
    bs=None
    for st,sa in itertools.product((0.1,0.2,0.3,0.4),(1,3,6)):
        pr=dict(best[1],st=st,sa=sa); res,o=scenA(tn,pr,"pool")
        sc=bar(o[5]["steer"]["new"],o[5]["steer"]["old"])-3*max(0,res["base"]["old"]-o[5]["steer"]["old"]-0.01)
        if bs is None or sc>bs[0]: bs=(sc,pr)
    return bs[1] if True else best[1]
# ============ SCENARIO B: systematic error correction (full-label prompt) ============
TASKS=["ag_news","dbpedia","yelp","amazon","emotion","offensive"]
def scenB(tn,params,split_eval):
    cal=ld("mdc_calib"); pool=ld("mdc_pool"); dev=ld("mdc_dev")
    def sub(z): m=z["task"]==tn; return {k:v[m] for k,v in z.items()}
    cal,pool,dev=sub(cal),sub(pool),sub(dev); n=int(cal["y"].max())+1
    head=Head(X(cal),cal["y"],r=params.get("r",96)); mus0=[head.proj(X(cal)[cal["y"]==c]).mean(0) for c in range(n)]; cnt0=[float((cal["y"]==c).sum()) for c in range(n)]
    ev=dev if split_eval=="dev" else pool
    Xe=X(ev); Le=lsm(ev["L"][:,:n]); ye=ev["y"]; Xp=X(pool); Lp=lsm(pool["L"][:,:n]); yp=pool["y"]
    lam0=params["lam"]; wrongp=np.where((Lp+lam0*head.score(Xp,mus0)).argmax(1)!=yp)[0]; wrongE=(Le+lam0*head.score(Xe,mus0)).argmax(1)!=ye   # "wrong" = wrong under the SHIPPED system (base + k=0 head)
    out={}
    for k in (0,)+KS:
        rows={m:[] for m in ("bias","lda_comb","combo","steer","lookup_ref")}
        for sd in range(SEEDS):
            rs=np.random.RandomState(sd); sel=rs.permutation(wrongp)[:k]
            if len(sel)<k: continue
            if k==0 and sd>0: continue
            mask=np.ones(len(ev["y"]),bool)
            if split_eval=="pool": mask[sel]=False
            def fin(pred): 
                m=mask&wrongE; mm=mask
                return dict(fix=float((pred[m]==ye[m]).mean()) if m.sum() else np.nan, acc=float((pred[mm]==ye[mm]).mean()),
                            corr_ok=float((pred[mm&~wrongE]==ye[mm&~wrongE]).mean()))
            ya=yp[sel]; pa=Lp[sel].argmax(1)
            # bias: prior-shift counts (true vs predicted among marks), eta strength
            t=np.bincount(ya,minlength=n)+params["ba"]; p=np.bincount(pa,minlength=n)+params["ba"]
            b=params["eta"]*np.log(t/p); rows["bias"].append(fin((Le+b).argmax(1)))
            if k==0: mus,cnt=mus0,cnt0
            else: mus,cnt=update(mus0,cnt0,head.proj(Xp[sel]),ya,params["w"])
            if k==0: b=np.zeros(n)
            S=head.score(Xe,mus); rows["lda_comb"].append(fin((Le+params["lam"]*S).argmax(1)))
            # steering in final-layer feature space (calib class means), v = mean(mu_true - mu_pred); ungated shift of prototype score
            # effect on logits through a calib-fit linear readout of the base logits (closed form, lstsq)
            rows["steer"].append(None)
            rows["combo"].append(fin((Le+params["lam"]*S+b).argmax(1)))
            # lookup reference (old design): nearest annotated hidden (cos on centred feats) >= tau -> stored label
            if k==0: rows["lookup_ref"].append(fin(Le.argmax(1))); continue
            Za=Xp[sel]-Xp.mean(0); Za/=np.linalg.norm(Za,axis=1,keepdims=True)+1e-9; Ze=Xe-Xp.mean(0); Ze/=np.linalg.norm(Ze,axis=1,keepdims=True)+1e-9
            Sm=Ze@Za.T; pr=Le.argmax(1).copy(); hit=Sm.max(1)>=params["tau"]; pr[hit]=ya[Sm.argmax(1)[hit]]; rows["lookup_ref"].append(fin(pr))
        rows.pop("steer"); out[k]={m:agg(r) for m,r in rows.items() if r}
    base=dict(fix=0.0,acc=float((Le.argmax(1)==ye).mean())); return base,out
def tuneB(tn):
    best=None
    for w,lam in itertools.product((3,10,30),(0.5,1.0,2.0)):
        pr=dict(w=w,lam=lam,eta=0.5,ba=1,tau=.9); b,o=scenB(tn,pr,"pool")
        sc=o[10]["lda_comb"]["fix"]+ (o[10]["lda_comb"]["corr_ok"]-1)*3
        if best is None or sc>best[0]: best=(sc,pr)
    pr=best[1]; bb=None
    for eta in (0.5,1.0,2.0):
        p2=dict(pr,eta=eta); b,o=scenB(tn,p2,"pool"); sc=o[10]["combo"]["fix"]+(o[10]["combo"]["corr_ok"]-1)*3
        if bb is None or sc>bb[0]: bb=(sc,p2)
    return bb[1]

# ============ SCENARIO C: MMLU (letter classes) ============
def scenC(params,split_eval):
    cal=ld("mmlu_calib"); pool=ld("mmlu_pool"); dev=ld("mmlu_dev")
    head=Head(X(cal),cal["y"],r=params.get("r",96)); mus0=[head.proj(X(cal)[cal["y"]==c]).mean(0) for c in range(4)]; cnt0=[float((cal["y"]==c).sum()) for c in range(4)]
    Xp=X(pool); Lp=lsm(pool["L"][:,:4]); yp=pool["y"]; sp=pool["subj"]
    lam0=params["lam"]; wrongp=np.where((Lp+lam0*head.score(Xp,mus0)).argmax(1)!=yp)[0]
    # eval pool: use second half of pool (never used for annotation), annotations from first half
    half=len(yp)//2; ann_ok=wrongp[wrongp<half]
    Xe,Le,ye,se=(X(dev),lsm(dev["L"][:,:4]),dev["y"],dev["subj"]) if split_eval=="dev" else (Xp[half:],Lp[half:],yp[half:],sp[half:])
    out={}
    for k in (0,)+KS:
        rows={m:[] for m in ("bias","lda_comb","combo","lookup_ref")}
        for sd in range(SEEDS):
            if k==0 and sd>0: continue
            sel=np.random.RandomState(sd).permutation(ann_ok)[:k]; ya=yp[sel]; pa=(Lp[sel]+lam0*head.score(Xp[sel],mus0)).argmax(1)
            subj=set(sp[sel]); rel=np.array([s_ in subj for s_ in se]); wrongE=(Le+lam0*head.score(Xe,mus0)).argmax(1)!=ye
            def fin(pred,Xq=None,predq=None):
                return dict(dev=float((pred==ye).mean()),rel=float((pred[rel]==ye[rel]).mean()) if rel.sum() else np.nan,
                            fix=float((pred[wrongE]==ye[wrongE]).mean()),rec=float(predq) if predq is not None else np.nan)
            t=np.bincount(ya,minlength=4)+params["ba"]; p=np.bincount(pa,minlength=4)+params["ba"]; b=params["eta"]*np.log(t/p)
            if k==0: mus=mus0
            else: mus,_=update(mus0,cnt0,head.proj(Xp[sel]),ya,params["w"])
            S=head.score(Xe,mus); Sr=head.score(Xp[sel],mus) if k else None
            rec=lambda sc: float((sc.argmax(1)==ya).mean()) if k else None
            rows["bias"].append(fin((Le+b).argmax(1),predq=rec(Lp[sel]+b)))
            rows["lda_comb"].append(fin((Le+params["lam"]*S).argmax(1),predq=rec(Lp[sel]+params["lam"]*Sr) if k else None))
            rows["combo"].append(fin((Le+params["lam"]*S+b).argmax(1),predq=rec(Lp[sel]+params["lam"]*Sr+b) if k else None))
            if k==0: rows["lookup_ref"].append(fin(Le.argmax(1))); continue
            Za=Xp[sel]-Xp.mean(0); Za/=np.linalg.norm(Za,axis=1,keepdims=True)+1e-9; Ze=Xe-Xp.mean(0); Ze/=np.linalg.norm(Ze,axis=1,keepdims=True)+1e-9
            Sm=Ze@Za.T; pr=Le.argmax(1).copy(); hit=Sm.max(1)>=params["tau"]; pr[hit]=ya[Sm.argmax(1)[hit]]; rows["lookup_ref"].append(fin(pr,predq=1.0))
        out[k]={m:agg(r) for m,r in rows.items()}
        out[k]["base"]=dict(dev=float((Le.argmax(1)==ye).mean()))
    return out
def tuneC():
    best=None
    for w,lam,eta in itertools.product((1,4),(0.1,0.3,1.0),(0.0,0.5)):
        pr=dict(w=w,lam=lam,eta=eta,ba=1,tau=.9,r=96); o=scenC(pr,"pool")
        sc=o[10]["combo"]["dev"]-o[10]["base"]["dev"]
        if best is None or sc>best[0]: best=(sc,pr)
    return best[1]
if __name__=="__main__":
    R=dict(A={},B={},C={})
    for tn in HOLD:
        pr=tuneA(tn); res,o=scenA(tn,pr,"dev"); R["A"][tn]=dict(params=pr,base=res["base"],byk=o)
        print("A",tn,pr,"base",{k:round(v,3) for k,v in res["base"].items()},flush=True)
        for k in KS: print("  k",k,{m:{q:round(v,3) for q,v in d.items() if q in("new","old","fa")} for m,d in o[k].items()},flush=True)
    for tn in TASKS:
        pr=tuneB(tn); b,o=scenB(tn,pr,"dev"); R["B"][tn]=dict(params=pr,base=b,byk=o)
        print("B",tn,pr,"base acc",round(b["acc"],3),flush=True)
        for k in (0,)+KS: print("  k",k,{m:{q:round(v,3) for q,v in d.items()} for m,d in o[k].items()},flush=True)
    pr=tuneC(); o=scenC(pr,"dev"); R["C"]=dict(params=pr,byk=o); print("C",pr,flush=True)
    for k in (0,)+KS: print("  k",k,{m:{q:round(v,3) for q,v in d.items()} for m,d in o[k].items()},flush=True)
    json.dump(R,open(f"results_{TAG}_L{LI}.json","w"),indent=1)
