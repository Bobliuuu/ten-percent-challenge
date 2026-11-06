# trackE: structured pruning + closed-form (ridge) compensation. NO gradient steps / training anywhere.
import torch, json, numpy as np, torch.nn.functional as F, math
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
M="textattack/bert-base-uncased-SST-2"
DEV="mps" if torch.backends.mps.is_available() else "cpu"
BASE_PARAMS=109483778
_cache={}
def data():
    if "d" in _cache: return _cache["d"]
    sp=json.load(open(__import__("os").path.exists("/root/splits.json") and "/root/splits.json" or "/Users/jhome/ten-percent/splits.json"))
    tr=load_dataset("nyu-mll/glue","sst2")["train"]   # TRAIN ONLY. validation/test never touched.
    tok=AutoTokenizer.from_pretrained(M)
    rng=np.random.RandomState(1)
    excl=set(sp["dev"])|set(sp["feedback"])
    others=[i for i in rng.permutation(len(tr)).tolist() if i not in excl][:4000]
    S=lambda idx:[tr[i]["sentence"] for i in idx]
    import sys; sys.path.insert(0,'/Users/jhome/ten-percent/trackG'); sys.path.insert(0,'/root/trackG'); from dev2 import load_dev2
    d2,l2=load_dev2(); h2,hl=load_dev2(hard=True)
    # calibrate on dev2[:2000] + hard[:750]; report on HELD-OUT dev2[2000:] (1000), hard[750:] (750), plus old SST-2 dev
    d=dict(tok=tok,tr=tr,dev=S(sp["dev"]),fb=S(sp["feedback"]),calib=d2[:2000]+h2[:750],
           d2=d2[2000:],l2=l2[2000:],h2=h2[750:],hl=hl[750:])
    _cache["d"]=d; return d
def load_base():
    m=AutoModelForSequenceClassification.from_pretrained(M).eval()
    sd={k:v.detach().clone().float() for k,v in m.state_dict().items()}
    return sd
def batches(texts,bs=128):
    tok=data()["tok"]
    order=np.argsort([len(t) for t in texts]); out=[]
    for i in range(0,len(texts),bs):
        ix=order[i:i+bs]; b=tok([texts[j] for j in ix],padding=True,truncation=True,max_length=64,return_tensors="pt")
        out.append((ix,b["input_ids"].to(DEV),b["attention_mask"].to(DEV)))
    return out
class Net:
    """functional BERT with per-layer pruned dims. emb: dict(word (V,768) or (U,Vt), vocab map)"""
    def __init__(s,sd):
        s.sd=sd; s.layers=[]
        g=lambda k:sd[k].to(DEV)
        s.pos=g("bert.embeddings.position_embeddings.weight"); s.typ=g("bert.embeddings.token_type_embeddings.weight")[0]
        s.eln=(g("bert.embeddings.LayerNorm.weight"),g("bert.embeddings.LayerNorm.bias"))
        s.word=g("bert.embeddings.word_embeddings.weight"); s.wfac=None; s.vmap=None
        s.pool=(g("bert.pooler.dense.weight"),g("bert.pooler.dense.bias")); s.cls=(g("classifier.weight"),g("classifier.bias"))
        for i in range(12):
            p=f"bert.encoder.layer.{i}."
            L=dict(src=i,nh=12,
              q=(g(p+"attention.self.query.weight"),g(p+"attention.self.query.bias")),
              k=(g(p+"attention.self.key.weight"),g(p+"attention.self.key.bias")),
              v=(g(p+"attention.self.value.weight"),g(p+"attention.self.value.bias")),
              o=(g(p+"attention.output.dense.weight"),g(p+"attention.output.dense.bias")),
              ln1=(g(p+"attention.output.LayerNorm.weight"),g(p+"attention.output.LayerNorm.bias")),
              w1=(g(p+"intermediate.dense.weight"),g(p+"intermediate.dense.bias")),
              w2=(g(p+"output.dense.weight"),g(p+"output.dense.bias")),
              ln2=(g(p+"output.LayerNorm.weight"),g(p+"output.LayerNorm.bias")))
            s.layers.append(L)
    def embed(s,ids):
        ids=ids if s.vmap is None else s.vmap[ids]
        w=s.word[ids] if s.wfac is None else s.word[ids]@s.wfac
        x=w+s.pos[:ids.shape[1]][None]+s.typ
        return F.layer_norm(x,(768,),*s.eln,eps=1e-12)
    def attn_ctx(s,L,x,mask):
        B,T,_=x.shape; nh=L["nh"]
        if nh==0: return None
        sp=lambda t:t.view(B,T,nh,64).transpose(1,2)
        q=sp(F.linear(x,*L["q"]));k=sp(F.linear(x,*L["k"]));v=sp(F.linear(x,*L["v"]))
        c=F.scaled_dot_product_attention(q,k,v,attn_mask=mask[:,None,None,:].bool())
        return c.transpose(1,2).reshape(B,T,nh*64)
    def layer_parts(s,L,x,mask):
        c=s.attn_ctx(L,x,mask)
        z1=x+(F.linear(c,*L["o"]) if c is not None else L["o"][1])
        a=F.layer_norm(z1,(768,),*L["ln1"],eps=1e-12)
        h=F.gelu(F.linear(a,*L["w1"])) if L["w1"][0].shape[0]>0 else None
        z2=a+(F.linear(h,*L["w2"]) if h is not None else L["w2"][1])
        return c,z1,a,h,z2,F.layer_norm(z2,(768,),*L["ln2"],eps=1e-12)
    def forward(s,ids,mask):
        x=s.embed(ids)
        for L in s.layers: x=s.layer_parts(L,x,mask)[-1]
        return F.linear(torch.tanh(F.linear(x[:,0],*s.pool)),*s.cls)
    def params(s):
        n=0
        if s.wfac is None: n+=s.word.numel()
        else: n+=s.word.numel()+s.wfac.numel()
        n+=s.typ.numel()+s.eln[0].numel()*2+s.pos.numel()
        for L in s.layers:
            for k in ("q","k","v","o","ln1","w1","w2","ln2"): n+=sum(t.numel() for t in L[k])
        n+=sum(t.numel() for t in s.pool)+sum(t.numel() for t in s.cls)
        return n
@torch.no_grad()
def predict(net,texts):
    out=np.zeros((len(texts),2),dtype=np.float32)
    for ix,ids,m in batches(texts): out[ix]=net.forward(ids,m).cpu().numpy()
    return out
def ridge(Fm,Y,lam):
    # Fm (T,k) float32 on DEV, Y (T,d). solves with bias, float64 on CPU. closed-form normal equations.
    T=Fm.shape[0]
    Fa=torch.cat([Fm,torch.ones(T,1,device=Fm.device)],1)
    G=(Fa.T@Fa).double().cpu(); R=(Fa.T@Y).double().cpu()
    dg=G.diag().clone(); k=G.shape[0]
    G+=lam*dg[:-1].mean()*torch.diag(torch.cat([torch.ones(k-1),torch.zeros(1)]).double())
    W=torch.linalg.solve(G,R)
    return W[:-1].T.float().to(DEV),W[-1].float().to(DEV)   # (d,k),(d,)
