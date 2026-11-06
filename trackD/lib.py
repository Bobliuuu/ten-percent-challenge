import torch, json, numpy as np, math, os
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
M="textattack/bert-base-uncased-SST-2"
BASE_BYTES=437958644
dev_dev=torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
_ds=None
def data():
    global _ds
    if _ds is None:
        sp=json.load(open(os.path.expanduser("~/ten-percent/splits.json")))
        tr=load_dataset("nyu-mll/glue","sst2")["train"]   # TRAIN only; validation/test never touched
        _ds=(tr.select(sp["dev"]),tr.select(sp["feedback"]))
    return _ds
tok=AutoTokenizer.from_pretrained(M)
def load():
    return AutoModelForSequenceClassification.from_pretrained(M).eval()
def batches(texts,bs=64):
    for i in range(0,len(texts),bs):
        yield tok(texts[i:i+bs],padding=True,truncation=True,max_length=64,return_tensors="pt")
@torch.no_grad()
def preds(m,texts,dev=None):
    dev=dev or dev_dev; m=m.to(dev); out=[]
    for b in batches(texts): out.append(m(**{k:v.to(dev) for k,v in b.items()}).logits.float().cpu())
    m.cpu(); return torch.cat(out)

def qparams(w,bits,sym=False):
    # w [rows, g] -> per-row scale (fp16-rounded), integer zero
    n=2**bits-1
    lo=w.min(1,keepdim=True)[0].clamp(max=0) if False else w.min(1,keepdim=True)[0]
    hi=w.max(1,keepdim=True)[0]
    s=((hi-lo).clamp(min=1e-8)/n).half().float().clamp(min=1e-8)
    z=(-lo/s).round().clamp(0,n)
    return s,z
def qd(w,s,z,bits):
    n=2**bits-1
    q=(w/s+z).round().clamp(0,n)
    return q,(q-z)*s

def rtn(W,bits,g):
    R,C=W.shape; Wg=W.reshape(R,C//g,g).reshape(-1,g)
    s,z=qparams(Wg,bits); q,d=qd(Wg,s,z,bits)
    return d.reshape(R,C),q.reshape(R,C).to(torch.int16),s.reshape(R,C//g),z.reshape(R,C//g)

def gptq(W,H,bits,g,damp=0.01,actorder=True,blk=128):
    W=W.clone().float(); R,C=W.shape
    H=H.clone(); dead=torch.diag(H)==0; H[dead,dead]=1; W[:,dead]=0
    if actorder:
        perm=torch.argsort(torch.diag(H),descending=True); W=W[:,perm]; H=H[perm][:,perm]
        invperm=torch.argsort(perm)
    H+=damp*torch.mean(torch.diag(H))*torch.eye(C)
    L=torch.linalg.cholesky(H); Hinv=torch.cholesky_inverse(L); Hinv=torch.linalg.cholesky(Hinv,upper=True)
    Q=torch.zeros_like(W); codes=torch.zeros(R,C,dtype=torch.int16)
    # groups are defined over the (possibly permuted) order; keep static groups in permuted order -> need perm stored (cheap: C*2B)
    S=[];Z=[]
    for i1 in range(0,C,blk):
        i2=min(i1+blk,C); W1=W[:,i1:i2].clone(); Q1=torch.zeros_like(W1); Err=torch.zeros_like(W1); Hi=Hinv[i1:i2,i1:i2]
        for i in range(i2-i1):
            c=i1+i
            if c%g==0:
                s,z=qparams(W[:,c:c+g] if i==0 else torch.cat([W1[:,i:],W[:,i2:]],1)[:,:g],bits); S.append(s);Z.append(z)
            w=W1[:,i:i+1]; q,d=qd(w,s,z,bits)
            Q1[:,i:i+1]=d; codes[:,c:c+1]=q.to(torch.int16)
            err=(w-d)/Hi[i,i]; W1[:,i:]-=err@Hi[i:i+1,i:]; Err[:,i:i+1]=err
        Q[:,i1:i2]=Q1; W[:,i2:]-=Err@Hinv[i1:i2,i2:]
    S=torch.cat(S,1);Z=torch.cat(Z,1)
    if actorder: Q=Q[:,invperm]
    return Q,codes,S,Z

def entropy_bits(codes,bits):
    # empirical order-0 entropy per tensor (ideal ANS cost); also per-column-group not modeled
    c=torch.bincount(codes.flatten().long(),minlength=2**bits).double(); p=c/c.sum(); p=p[p>0]
    return float(-(p*p.log2()).sum()*codes.numel())
