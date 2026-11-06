import sys,os,json,struct,math,time,lzma,zlib
import numpy as np, torch
_R=os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),'..')); sys.path.insert(0,_R+'/trackA'); sys.path.insert(0,_R+'/trackG')
from lib import tok,dev_data,feedback_data,base_model,quant_rtn,dequant,encode,prune_vocab,build_vocab,train_ds
DEV='cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
@torch.no_grad()
def logits(m,batches,dev=None):
    dev=dev or DEV; m=m.to(dev); out=[]
    for ids,tt,am in batches: out.append(m(input_ids=ids.to(dev),token_type_ids=tt.to(dev),attention_mask=am.to(dev)).logits.float().cpu())
    return torch.cat(out)
from dev2 import load_dev2
HERE=os.path.dirname(os.path.abspath(__file__))
LIN=["attention.self.query","attention.self.key","attention.self.value","attention.output.dense","intermediate.dense","output.dense"]
SHORT=dict(zip(LIN,["q","k","v","a","i","o"]))
BUDGET=43795864
ID=torch.arange(tok.vocab_size)
def enc(texts): return encode(texts,ID)
_cache={}
def data():
    if not _cache:
        t2,l2=load_dev2(); th,lh=load_dev2(hard=True)
        txd,_=dev_data()
        _cache.update(dev=txd,d2=t2,dh=th,
            B_cal=enc(t2[:512]),B_ev=enc(t2[512:]),B_h=enc(th),B_dev=enc(txd[:1000]))
    return _cache
def fullbase():
    m=prune_vocab(base_model(),list(range(tok.vocab_size)),64); return m
def kl_agree(lq,lb):
    pq=lq.log_softmax(-1); pb=lb.log_softmax(-1)
    kl=(pb.exp()*(pb-pq)).sum(-1).mean().item(); ag=(lq.argmax(-1)==lb.argmax(-1)).float().mean().item(); return kl,ag
# ---------- quantizers
def quant_clip(w,bits,g,fs=(1.0,.95,.9,.85,.8,.75,.7,.65,.6)):
    """RTN minmax with per-group MSE clip search; returns q,s,z (fp16-rounded)"""
    O,I=w.shape; wg=w.reshape(O,I//g,g); mn=wg.amin(-1); mx=wg.amax(-1); qm=2**bits-1; best=None
    for f in fs:
        c=(mn+mx)/2; lo=c-(c-mn)*f; hi=c+(mx-c)*f   # shrink about the center
        s=((hi-lo)/qm).clamp(min=1e-8).half().float(); z=(-lo/s).round().clamp(0,qm).half().float()
        q=(wg/s[...,None]+z[...,None]).round().clamp(0,qm); e=(((q-z[...,None])*s[...,None]-wg)**2).sum(-1)
        if best is None: best=[e,s,z,q]
        else:
            m=e<best[0]
            for j,t in enumerate((e,s,z,q)): best[j]=torch.where(m if j<3 else m[...,None],t,best[j])
    return best[3].reshape(O,I),best[1],best[2]
def gptq2(W,H,bits,g,blocksize=128,damp=0.01,clip=True,order=None):
    """GPTQ, groupwise asym. order: optional column permutation (act-order with STATIC groups on original index)."""
    dv=W.device; H=H.to(dv)
    W=W.clone().float(); O,I=W.shape; qm=2**bits-1
    H=H.clone(); dead=torch.diag(H)==0; H[dead,dead]=1; W[:,dead]=0
    # static group grids from original weights (with MSE clip) -- identical to dynamic when no order
    if order is not None:
        W=W[:,order]; H=H[order][:,order]
    H+=damp*torch.mean(torch.diag(H))*torch.eye(I,device=dv)
    L=torch.linalg.cholesky(H); Hinv=torch.linalg.cholesky(torch.cholesky_inverse(L),upper=True)
    Q=torch.zeros_like(W); QI=torch.zeros_like(W); ng=I//g
    S=torch.zeros(O,ng,device=dv); Z=torch.zeros(O,ng,device=dv)
    if order is not None:
        _,S0,Z0=quant_clip(W[:,torch.argsort(order)],bits,g) if clip else quant_rtn(W[:,torch.argsort(order)],bits,g)
        gid=(order//g)
    for i1 in range(0,I,blocksize):
        i2=min(i1+blocksize,I); W1=W[:,i1:i2].clone(); Err=torch.zeros_like(W1); Hi=Hinv[i1:i2,i1:i2]
        for i in range(i2-i1):
            c=i1+i
            if order is None:
                if c%g==0:
                    wg=W[:,c:c+g] if i==0 else torch.cat([W1[:,i:],W[:,i2:]],1)[:,:g]
                    mn=wg.amin(1); mx=wg.amax(1); best=None
                    for f in ((1.0,.95,.9,.85,.8,.75,.7,.65,.6) if clip else (1.0,)):
                        cc=(mn+mx)/2; lo=cc-(cc-mn)*f; hi=cc+(mx-cc)*f
                        s2=((hi-lo)/qm).clamp(min=1e-8).half().float(); z2=(-lo/s2).round().clamp(0,qm).half().float()
                        e=(((wg/s2[:,None]+z2[:,None]).round().clamp(0,qm)-z2[:,None])*s2[:,None]-wg).pow(2).sum(1)
                        if best is None: best=(e,s2,z2)
                        else:
                            m_=e<best[0]; best=(torch.where(m_,e,best[0]),torch.where(m_,s2,best[1]),torch.where(m_,z2,best[2]))
                    S[:,c//g]=best[1]; Z[:,c//g]=best[2]
                s=S[:,c//g]; z=Z[:,c//g]
            else:
                gi=int(gid[c]); s=S0[:,gi]; z=Z0[:,gi]
            w=W1[:,i]; q=(w/s+z).round().clamp(0,qm); QI[:,c]=q; dq=(q-z)*s; Q[:,c]=dq
            err=(w-dq)/Hi[i,i]; W1[:,i:]-=err.unsqueeze(1)*Hi[i,i:].unsqueeze(0); Err[:,i]=err
        W[:,i2:]-=Err@Hinv[i1:i2,i2:]
    if order is not None:
        inv=torch.argsort(order); QI=QI[:,inv]; S,Z=S0,Z0
    return QI,S,Z
# ---------- entropy size
def ent_bytes(a,k):
    c=np.bincount(np.asarray(a).ravel().astype(np.int64),minlength=k).astype(float); p=c[c>0]/c.sum(); return float(-(p*np.log2(p)).sum()*c.sum()/8)
def est_size(q,z,bits):
    return ent_bytes(q.numpy(),2**bits)+ent_bytes(z.numpy(),2**bits)+2*z.numel()+ 2**bits*3
# ---------- hessians
@torch.no_grad()
def collect_H(m,batches):
    """independent fp Hessians for all encoder linears + pooler input; returns dict key->H (cpu), count"""
    m=m.to(DEV).eval(); Hs={}; cur={}; hs=[]; cnt=[0]
    def mk(key,cls=False):
        def hook(mod,inp):
            x=inp[0].float()
            if cls: x=x.reshape(-1,x.shape[-1])
            else: x=x[cur["m"].bool()]
            Hs[key]=Hs.get(key,0)+(x.T@x)
        return hook
    for l,L in enumerate(m.bert.encoder.layer):
        for n in LIN: hs.append(L.get_submodule(n).register_forward_pre_hook(mk(f"L{l}.{n}")))
    hs.append(m.bert.pooler.dense.register_forward_pre_hook(mk("pool",True)))
    for ids,tt,am in batches:
        cur["m"]=am.to(DEV); cnt[0]+=int(am.sum()); m(input_ids=ids.to(DEV),token_type_ids=tt.to(DEV),attention_mask=am.to(DEV))
    for h in hs: h.remove()
    return {k:(2*v/cnt[0]).cpu() for k,v in Hs.items()}
def emb_groups(nb=5):
    """token frequency groups from SST-2 train text; returns list of index tensors"""
    f=HERE+"/tokfreq.npy"
    if os.path.exists(f): c=np.load(f)
    else:
        c=np.zeros(tok.vocab_size,dtype=np.int64); s=train_ds()["sentence"]
        for i in range(0,len(s),2000):
            for x in tok(s[i:i+2000],truncation=True,max_length=64)["input_ids"]: np.add.at(c,x,1)
        np.save(f,c)
    edges=[(0,1),(1,3),(3,10),(10,100),(100,10**9)]
    return [torch.from_numpy(np.where((c>=a)&(c<b))[0]) for a,b in edges],c
# ---------- trackK-style Lloyd-Max embedding quantizer (global non-uniform levels, per-group fp16 RMS scale, mean-row centered)
def lm_levels(x,bits,iters=40):
    k=2**bits; flat=x.reshape(-1); gen=torch.Generator().manual_seed(0)
    smp=flat[torch.randint(0,flat.numel(),(min(1500000,flat.numel()),),generator=gen)]
    lv=torch.quantile(smp,(torch.arange(k)+0.5)/k)
    for _ in range(iters):
        a=(smp[:,None]-lv[None]).abs().argmin(1)
        for j in range(k):
            m=a==j
            if m.any(): lv[j]=smp[m].mean()
    return lv.half().float().sort()[0]
def quant_lm(W,bits,g,mu):
    N,D=W.shape; Wg=(W-mu).reshape(N,D//g,g); sc=Wg.pow(2).mean(2,keepdim=True).sqrt().half().float().clamp(min=1e-8)
    x=Wg/sc; lv=lm_levels(x,bits)
    q=torch.cat([(c[:,None]-lv[None]).abs().argmin(1) for c in x.reshape(-1).split(2000000)]).reshape(N,D)
    return q,sc.reshape(N,-1),lv
def dequant_lm(q,sc,lv,g,mu):
    N,D=q.shape; return (lv[q].reshape(N,D//g,g)*sc.reshape(N,D//g,1)).reshape(N,D)+mu
def emb_tiers(sizes=(2000,4000,5500)):
    c=np.load(HERE+"/tokfreq.npy") if os.path.exists(HERE+"/tokfreq.npy") else emb_groups()[1]
    special=[0,100,101,102,103]; order=np.lexsort((np.arange(len(c)),-c)); order=np.array(special+[i for i in order if i not in special])
    out=[];p=0
    for n in sizes: out.append(torch.from_numpy(np.sort(order[p:p+n]))); p+=n
    out.append(torch.from_numpy(np.sort(order[p:]))); return out

def rot_mat(C,name,blk=256):
    import fmt; return fmt.rot_mat(C,name,blk)
