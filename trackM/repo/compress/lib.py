import torch, json, numpy as np, math, copy, os, sys
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
M="textattack/bert-base-uncased-SST-2"
DEV="mps" if torch.backends.mps.is_available() else "cpu"
HERE=os.path.dirname(os.path.abspath(__file__))
tok=AutoTokenizer.from_pretrained(M)
_ds=None
def train_ds():
    global _ds
    if _ds is None: _ds=load_dataset("nyu-mll/glue","sst2")["train"]   # ONLY train split is ever touched
    return _ds
def dev_data():
    sp=json.load(open(HERE+"/../splits.json"))["dev"]
    d=train_ds().select(sp); return d["sentence"],d["label"]
def feedback_data():
    sp=json.load(open(HERE+"/../splits.json"))["feedback"]
    d=train_ds().select(sp); return d["sentence"],d["label"]
def base_model():
    return AutoModelForSequenceClassification.from_pretrained(M).eval()
# ---- vocab pruning (train text only)
def build_vocab():
    f=HERE+"/vocab_keep.json"
    if os.path.exists(f): return json.load(open(f))
    ids=set(tok.all_special_ids)
    s=train_ds()["sentence"]
    for i in range(0,len(s),2000):
        for x in tok(s[i:i+2000],truncation=True,max_length=64)["input_ids"]: ids.update(x)
    keep=sorted(ids); json.dump(keep,open(f,"w")); return keep
def remap_table(keep):
    t=torch.full((tok.vocab_size,),tok.unk_token_id,dtype=torch.long)
    for n,o in enumerate(keep): t[o]=n
    # new ids: unk id new
    t[t==tok.unk_token_id]=keep.index(tok.unk_token_id)
    return t
def encode(texts,table,maxlen=64):
    out=[]
    for i in range(0,len(texts),128):
        b=tok(texts[i:i+128],padding=True,truncation=True,max_length=maxlen,return_tensors="pt")
        out.append((table[b["input_ids"]],b["token_type_ids"],b["attention_mask"]))
    return out
def prune_vocab(m,keep,npos=None):
    e=m.bert.embeddings
    w=e.word_embeddings.weight.data[torch.tensor(keep)].clone()
    e.word_embeddings=torch.nn.Embedding(len(keep),w.shape[1]); e.word_embeddings.weight.data=w
    m.config.vocab_size=len(keep)
    if npos:
        e.position_embeddings.weight.data=e.position_embeddings.weight.data[:npos].clone()
        e.position_embeddings.num_embeddings=npos
        e.position_ids=torch.arange(npos).unsqueeze(0) if hasattr(e,"position_ids") else None
        if hasattr(e,"position_ids") and e.position_ids is not None: e.register_buffer("position_ids",torch.arange(npos).unsqueeze(0),persistent=False)
        if hasattr(e,"token_type_ids"): e.register_buffer("token_type_ids",torch.zeros(1,npos,dtype=torch.long),persistent=False)
    return m
@torch.no_grad()
def logits(m,batches,dev=DEV):
    m=m.to(dev); out=[]
    for ids,tt,am in batches:
        out.append(m(input_ids=ids.to(dev),token_type_ids=tt.to(dev),attention_mask=am.to(dev)).logits.float().cpu())
    return torch.cat(out)
# ---- quantization
def quant_rtn(w,bits,g):
    """asymmetric min-max groupwise along last dim; returns (q int, scale, zero) with w ~ (q-z)*s ; q in [0,2^b-1]"""
    O,I=w.shape; gg=I if g<=0 else g
    assert I%gg==0
    wg=w.reshape(O,I//gg,gg)
    mn=wg.amin(-1,keepdim=True).clamp(max=0)*0+wg.amin(-1,keepdim=True); mx=wg.amax(-1,keepdim=True)
    qm=2**bits-1
    s=((mx-mn)/qm).clamp(min=1e-8).half().float()
    z=(-mn/s).round().clamp(0,qm).half().float()
    q=(wg/s+z).round().clamp(0,qm)
    return q.reshape(O,I),s.squeeze(-1),z.squeeze(-1)
def dequant(q,s,z,g):
    O,I=q.shape; gg=I if g<=0 else g
    return ((q.reshape(O,I//gg,gg)-z.unsqueeze(-1))*s.unsqueeze(-1)).reshape(O,I)
def gptq(W,H,bits,g,blocksize=128,damp=0.01,clip=False):
    """GPTQ with groupwise asym quant; W (O,I). returns q,s,z (grids fixed from updated weights at group start)."""
    W=W.clone().float(); O,I=W.shape; gg=I if g<=0 else g
    H=H.clone(); dead=torch.diag(H)==0; H[dead,dead]=1; W[:,dead]=0
    H+=damp*torch.mean(torch.diag(H))*torch.eye(I)
    L=torch.linalg.cholesky(H); Hinv=torch.cholesky_inverse(L); Hinv=torch.linalg.cholesky(Hinv,upper=True)
    qm=2**bits-1
    Q=torch.zeros_like(W); S=torch.zeros(O,I//gg); Z=torch.zeros(O,I//gg); QI=torch.zeros_like(W)
    for i1 in range(0,I,blocksize):
        i2=min(i1+blocksize,I); W1=W[:,i1:i2].clone(); Err=torch.zeros_like(W1); Hi=Hinv[i1:i2,i1:i2]
        for i in range(i2-i1):
            c=i1+i
            if c%gg==0:
                wg=W[:,c:c+gg] if i==0 else torch.cat([W1[:,i:],W[:,i2:]],1)[:,:gg]
                mn=wg.amin(1); mx=wg.amax(1)
                s=((mx-mn)/qm).clamp(min=1e-8).half().float(); z=(-mn/s).round().clamp(0,qm).half().float()
                if clip:
                    best=None
                    for f in [1.0,0.95,0.9,0.85,0.8,0.75,0.7]:
                        s2=((mx-mn)*f/qm).clamp(min=1e-8).half().float(); z2=(-(mn*f)/s2).round().clamp(0,qm).half().float()
                        e=(((wg/s2[:,None]+z2[:,None]).round().clamp(0,qm)-z2[:,None])*s2[:,None]-wg).pow(2).sum(1)
                        if best is None: best=(e,s2,z2)
                        else:
                            m_=e<best[0]; best=(torch.where(m_,e,best[0]),torch.where(m_,s2,best[1]),torch.where(m_,z2,best[2]))
                    s,z=best[1],best[2]
                S[:,c//gg]=s; Z[:,c//gg]=z
            s=S[:,c//gg]; z=Z[:,c//gg]
            w=W1[:,i]; q=(w/s+z).round().clamp(0,qm); QI[:,c]=q
            dq=(q-z)*s; Q[:,c]=dq
            err=(w-dq)/Hi[i,i]; W1[:,i:]-=err.unsqueeze(1)*Hi[i,i:].unsqueeze(0); Err[:,i]=err
        W[:,i2:]-=Err@Hinv[i1:i2,i2:]
    return QI,S,Z
