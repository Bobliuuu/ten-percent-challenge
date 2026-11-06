"""trackJ: calibration-only corrections after low-bit weight quantization of BERT SST-2 encoder.
Usage: python tj.py '[{"name":"x","bits":3,"g":64,"method":"gptq", ...flags}]'
flags: method rtn|gptq ; clip(bool) ; bias ; scale ; ln ; rot(block size,0=off) ; awq ; cle ; ncal
Encoder = 12 layers x (q,k,v,o,f1,f2) + pooler. Embeddings/LN/biases/classifier stay fp16 (not counted: other tracks)."""
import sys, json, time, math, copy, torch, numpy as np
sys.path.insert(0,__import__('os').environ.get('D2DIR','/Users/jhome/ten-percent/trackG')); from dev2 import load_dev2
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.linalg import hadamard
torch.set_num_threads(int(__import__('os').environ.get('NT','3')))
M="textattack/bert-base-uncased-SST-2"
DEV=torch.device("cuda") if torch.cuda.is_available() else (torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu"))
WD=DEV if DEV.type=="cuda" else torch.device("cpu")   # device for GPTQ/closed-form math
import os
D2PATH=os.environ.get("D2DIR")
KEEP=os.environ.get("KEEP","/Users/jhome/ten-percent/trackA/vocab_keep.json")
tok=AutoTokenizer.from_pretrained(M)
def batches(texts,bs=64):
    for i in range(0,len(texts),bs):
        yield {k:v.to(DEV) for k,v in tok(texts[i:i+bs],padding=True,truncation=True,max_length=64,return_tensors="pt").items()}
@torch.no_grad()
def preds(m,texts):
    return torch.cat([m(**b).logits.float().cpu() for b in batches(texts)])

# ---------- quantizer primitives
def qparams(w,bits,clip=False):
    n=2**bits-1; lo=w.min(1,keepdim=True)[0]; hi=w.max(1,keepdim=True)[0]
    def mk(lo,hi):
        s=((hi-lo).clamp(min=1e-8)/n).half().float().clamp(min=1e-8); z=(-lo/s).round().clamp(0,n); return s,z
    s,z=mk(lo,hi)
    if not clip: return s,z
    def err(s,z): return (((w/s+z).round().clamp(0,n)-z)*s-w).pow(2).sum(1,keepdim=True)
    be=err(s,z)
    for p in [0.97,0.94,0.9,0.86,0.82,0.78,0.74,0.7]:
        s2,z2=mk(lo*p,hi*p); e=err(s2,z2); m=e<be
        s=torch.where(m,s2,s); z=torch.where(m,z2,z); be=torch.where(m,e,be)
    return s,z
def qd(w,s,z,bits):
    q=(w/s+z).round().clamp(0,2**bits-1); return q,(q-z)*s

def rtn(W,bits,g,clip=False):
    R,C=W.shape; Wg=W.reshape(-1,g); s,z=qparams(Wg,bits,clip); q,d=qd(Wg,s,z,bits)
    return d.reshape(R,C),q.reshape(R,C),s.numel()

def prep(H,W,damp=0.01):
    H=H.clone(); dead=torch.diag(H)==0; H[dead,dead]=1; W=W.clone(); W[:,dead]=0
    H+=damp*torch.mean(torch.diag(H))*torch.eye(H.shape[0],device=H.device)
    return W,torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(H)),upper=True)

def gptq(W,H,bits,g,clip=False,blk=128,actorder=True):
    R,C=W.shape; perm=torch.argsort(torch.diag(H),descending=True) if actorder else torch.arange(C,device=W.device)
    inv=torch.argsort(perm); W=W[:,perm]; H=H[perm][:,perm]; W,Hinv=prep(H,W)
    Q=torch.zeros_like(W); codes=torch.zeros(R,C,device=W.device); ng=0
    for i1 in range(0,C,blk):
        i2=min(i1+blk,C); W1=W[:,i1:i2].clone(); Q1=torch.zeros_like(W1); Err=torch.zeros_like(W1); Hi=Hinv[i1:i2,i1:i2]
        for i in range(i2-i1):
            c=i1+i
            if c%g==0: s,z=qparams(W1[:,i:i+g],bits,clip); ng+=R
            w=W1[:,i:i+1]; q,d=qd(w,s,z,bits); Q1[:,i:i+1]=d; codes[:,c:c+1]=q
            err=(w-d)/Hi[i,i]; W1[:,i:]-=err@Hi[i:i+1,i:]; Err[:,i:i+1]=err
        Q[:,i1:i2]=Q1; W[:,i2:]-=Err@Hinv[i1:i2,i2:]
    return Q[:,inv],codes[:,inv],ng

def had_rot(C,blk,seed):
    gen=torch.Generator().manual_seed(seed); sg=(torch.randint(0,2,(C,),generator=gen)*2-1).float()
    Hb=torch.tensor(hadamard(blk),dtype=torch.float32)/math.sqrt(blk)
    Rm=torch.zeros(C,C)
    for i in range(0,C,blk): Rm[i:i+blk,i:i+blk]=Hb
    return sg[:,None]*Rm   # R = D*blockHad (orthogonal)

# ---------- capture helpers
class Cap:
    def __init__(s,model,mods,kind):
        s.buf=[None]*len(mods); s.hs=[]; s.mask=None
        for i,mod in enumerate(mods):
            s.hs.append(mod.register_forward_hook(lambda m,inp,out,i=i:s.buf.__setitem__(i,(inp[0] if kind=='in' else out).detach().float())))
    def get(s,i,mask):
        x=s.buf[i].reshape(-1,s.buf[i].shape[-1]); return x[mask] if x.shape[0]==mask.numel() else x
    def rm(s): [h.remove() for h in s.hs]

def run(cfg):
    torch.manual_seed(0); t0=time.time()
    bits=cfg["bits"]; g=cfg["g"]; method=cfg.get("method","gptq"); bf=cfg.get("bits_fn")
    t2,y2=load_dev2(); th,_=load_dev2(hard=True); cal=t2[:cfg.get("ncal",512)]; ev=t2[512:]
    m0=AutoModelForSequenceClassification.from_pretrained(M).eval().to(DEV)
    mq=copy.deepcopy(m0)
    for p in list(m0.parameters())+list(mq.parameters()): p.requires_grad_(False)
    b_ev=preds(m0,ev); b_h=preds(m0,th)
    cb=list(batches(cal,64))
    def L(m,i): return m.bert.encoder.layer[i]
    tot=dict(bits=0.,w=0,extra=0.)
    # ---- cross-layer equalization (exact folds, weight-only)
    if cfg.get("cle"):
        for i in range(12):
            l=mq.bert.encoder.layer[i]; at=l.attention.self; o=l.attention.output.dense; nh=12; hd=64
            with torch.no_grad():
                # v rows (per output channel c) <-> o cols (c)
                rv=at.value.weight.abs().max(1)[0]; ro=o.weight.abs().max(0)[0]
                s=(rv/ro).sqrt().clamp(0.1,10) ; s=s/s.mean() if False else s
                at.value.weight/=s[:,None]; at.value.bias/=s; o.weight*=s[None,:]
                # q rows <-> k rows (same channel within head): q*=t, k/=t
                rq=at.query.weight.abs().max(1)[0]; rk=at.key.weight.abs().max(1)[0]
                t=(rk/rq).sqrt().clamp(0.1,10)
                at.query.weight*=t[:,None]; at.query.bias*=t; at.key.weight/=t[:,None]; at.key.bias/=t
    if cfg.get("recipe")=="A":
        keep=torch.tensor(json.load(open(KEEP)),device=WD); E=mq.bert.embeddings.word_embeddings.weight.data.to(WD)
        msk=torch.zeros(E.shape[0],dtype=torch.bool,device=WD); msk[keep]=True; out=E.clone()
        out[msk]=rtn(E[msk],3,128,True)[0]; out[~msk]=rtn(E[~msk],2,128,True)[0]; mq.bert.embeddings.word_embeddings.weight.data=out.to(DEV)
    # sanity after CLE
    stages=[]
    for i in range(12):
        l=L(mq,i); l0=L(m0,i)
        stages+= [(f"l{i}.qkv",[(l.attention.self.query,l0.attention.self.query),(l.attention.self.key,l0.attention.self.key),(l.attention.self.value,l0.attention.self.value)],None),
                  (f"l{i}.o",[(l.attention.output.dense,l0.attention.output.dense)],(l.attention.output.LayerNorm,l0.attention.output.LayerNorm)),
                  (f"l{i}.f1",[(l.intermediate.dense,l0.intermediate.dense)],None),
                  (f"l{i}.f2",[(l.output.dense,l0.output.dense)],(l.output.LayerNorm,l0.output.LayerNorm))]
    stages.append(("pool",[(mq.bert.pooler.dense,m0.bert.pooler.dense)],None))
    need_h = method=="gptq" or cfg.get("awq") or cfg.get("rot")
    for name,pairs,ln in stages:
        mods=[p[0] for p in pairs]; mods0=[p[1] for p in pairs]
        bqs=[bits]*len(mods); gs=[g]*len(mods)
        if cfg.get("recipe")=="A":
            li=int(name[1:].split('.')[0]) if name[0]=='l' else 0; k=name.split('.')[-1]
            bqs={'qkv':[2,2,3],'o':[3],'f1':[4 if li<6 else 3],'f2':[3],'pool':[6]}[k]; gs={'qkv':[128]*3,'o':[128],'f1':[128],'f2':[256],'pool':[128]}[k]
        bq=bqs[0]
        if cfg.get("skip") and name in cfg["skip"]: continue
        C=mods[0].in_features
        Hx=torch.zeros(C,C,device=WD); am=torch.zeros(C,device=WD); nt=0; samp=[]
        if need_h:
            cp=Cap(mq,mods[:1],'in')
            with torch.no_grad():
                for b in cb:
                    mq(**b); mk=b["attention_mask"].reshape(-1).bool(); x=cp.get(0,mk).to(WD)
                    Hx+=x.T@x; am+=x.abs().sum(0); nt+=x.shape[0]
                    if sum(s_.shape[0] for s_ in samp)<3072: samp.append(x[:512])
            cp.rm(); Hx/=nt; am/=nt
        # transform T: x'=Tinv x, W'=W Tfwd
        sdiag=torch.ones(C,device=WD)
        if cfg.get("awq"):
            X=torch.cat(samp)[:3072]; best=None
            for a in [0,0.25,0.5,0.75,1.0]:
                sc=(am.clamp(min=1e-5)**a); sc=sc/sc.mean()
                e=0.
                for mod in mods:
                    W=mod.weight.data.to(WD); Wq,_,_=rtn(W*sc[None,:],bq,g,cfg.get("clip",False))
                    e+=float((((Wq/sc[None,:])-W)@X.T).pow(2).mean())
                if best is None or e<best[0]: best=(e,sc,a)
            sdiag=best[1]; tot["extra"]+=C*16
        Rm=None
        if cfg.get("rot"): Rm=had_rot(C,cfg["rot"],sum(map(ord,name))%1000).to(WD)
        # Tfwd=diag(s) R ; Tinv=R^T diag(1/s)
        Tfwd=torch.diag(sdiag) if Rm is None else sdiag[:,None]*Rm
        Tinv=torch.diag(1/sdiag) if Rm is None else Rm.T*(1/sdiag)[None,:]
        trivial= Rm is None and not cfg.get("awq")
        Hp=Hx if trivial else Tinv@Hx@Tinv.T
        first=True
        for mi,mod in enumerate(mods):
            bq=bqs[mi]; g=gs[mi]
            W=mod.weight.data.to(WD); Wp=W if trivial else W@Tfwd
            if name=="pool": d,codes,ng=rtn(W,bq,g,cfg.get("clip",False)); mod.weight.data=d.to(DEV); continue
            if method=="rtn": d,codes,ng=rtn(Wp,bq,g,cfg.get("clip",False))
            elif method=="gptq": d,codes,ng=gptq(Wp,Hp,bq,g,cfg.get("clip",False),actorder=cfg.get("actorder",True))
            Weff=d if trivial else d@Tinv
            mod.weight.data=Weff.to(DEV)
            tot["bits"]+=W.numel()*bq+ng*(16+bq); tot["w"]+=W.numel()
        if method!="rtn" and cfg.get("actorder",True): tot["extra"]+=C*math.ceil(math.log2(C))
        # ---- output-statistics corrections (closed form; fold into group scales / biases / LN gain-beta)
        if cfg.get("bias") or cfg.get("scale"):
            targets=[(mod,mod0,'lin') for mod,mod0 in pairs]
            fix(mq,m0,cb,targets,cfg)
        if ln is not None and cfg.get("ln"):
            fix(mq,m0,cb,[(ln[0],ln[1],'ln')],cfg)
    pe=preds(mq,ev); ph=preds(mq,th)
    a2=float((pe.argmax(-1)==b_ev.argmax(-1)).float().mean()); ah=float((ph.argmax(-1)==b_h.argmax(-1)).float().mean())
    mad=float((pe-b_ev).abs().mean())
    ebits=(tot["bits"]+tot["extra"])/tot["w"]
    return dict(name=cfg["name"],bits_per_w=round(ebits,4),agree2=round(a2,4),agree_hard=round(ah,4),mean_abs_logit_err=round(mad,4),
                acc2=round(float((pe.argmax(-1)==torch.tensor(y2[512:])).float().mean()),4),sec=round(time.time()-t0))

def fix(mq,m0,cb,targets,cfg):
    """per-channel y' = a*y + b matching fp model output (a from regression if cfg.scale else 1)"""
    nT=len(targets); st=[None]*nT
    cq=Cap(mq,[t[0] for t in targets],'out'); cf=Cap(m0,[t[1] for t in targets],'out')
    with torch.no_grad():
        for b in cb:
            mq(**b); m0(**b); mk=b["attention_mask"].reshape(-1).bool()
            for i in range(nT):
                yq=cq.get(i,mk).to(WD).double(); yf=cf.get(i,mk).to(WD).double()
                s=[len(yq),yq.sum(0),yf.sum(0),(yq*yq).sum(0),(yq*yf).sum(0)]
                st[i]=s if st[i] is None else [u+v for u,v in zip(st[i],s)]
    cq.rm(); cf.rm()
    for (mod,_,kind),(n,sq,sf,sqq,sqf) in zip(targets,st):
        mq_=sq/n; mf=sf/n; var=sqq/n-mq_**2; cov=sqf/n-mq_*mf
        a=(cov/var.clamp(min=1e-8)).clamp(0.7,1.4) if cfg.get("scale") else torch.ones_like(mq_)
        bb=(mf-a*mq_)
        a=a.float().to(DEV); bb=bb.float().to(DEV)
        with torch.no_grad():
            if kind=='lin':
                mod.weight.mul_(a[:,None]); mod.bias.mul_(a).add_(bb)
            else:
                mod.weight.mul_(a); mod.bias.mul_(a).add_(bb)

if __name__=="__main__":
    for c in json.loads(sys.argv[1]):
        r=run(c); r["cfg"]={k:v for k,v in c.items() if k!="name"}; print(json.dumps(r),flush=True)
