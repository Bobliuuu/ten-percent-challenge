# Training-free quantization engine (GPU). cfg-driven; see RESULTS.md
import json, heapq, math, torch, numpy as np
M="textattack/bert-base-uncased-SST-2"; BASE_BYTES=437958644
D="cuda" if torch.cuda.is_available() else "cpu"

def huff_bits(codes,nsym):
    c=torch.bincount(codes.flatten().long().cpu(),minlength=nsym).tolist(); cnt=sum(c)
    h=[(x,i,None) for i,x in enumerate(c) if x>0]
    if len(h)==1: return cnt
    heapq.heapify(h); uid=nsym; depth={}
    nodes={}
    while len(h)>1:
        a=heapq.heappop(h); b=heapq.heappop(h); uid+=1; nodes[uid]=(a[1],b[1]); heapq.heappush(h,(a[0]+b[0],uid,None))
    L={}
    st=[(h[0][1],0)]
    while st:
        n,d=st.pop()
        if n in nodes: st+= [(nodes[n][0],d+1),(nodes[n][1],d+1)]
        else: L[n]=d
    return sum(c[i]*L[i] for i in L)
def ent_bits(codes,nsym):
    c=torch.bincount(codes.flatten().long().cpu(),minlength=nsym).double(); p=c/c.sum(); p=p[p>0]
    return float(-(p*p.log2()).sum()*codes.numel())

def qparams(w,bits):
    n=2**bits-1; lo=w.min(1,keepdim=True)[0]; hi=w.max(1,keepdim=True)[0]
    s=((hi-lo).clamp(min=1e-8)/n).half().float().clamp(min=1e-8); z=(-lo/s).round().clamp(0,n); return s,z
def qd(w,s,z,bits):
    q=(w/s+z).round().clamp(0,2**bits-1); return q,(q-z)*s
def rtn(W,bits,g):
    R,C=W.shape; Wg=W.reshape(-1,g); s,z=qparams(Wg,bits); q,d=qd(Wg,s,z,bits)
    return d.reshape(R,C),q.reshape(R,C),s.numel()*(16+bits)

def prep_H(H,W,damp=0.01):
    H=H.clone(); dead=torch.diag(H)==0; H[dead,dead]=1; W=W.clone(); W[:,dead]=0
    H+=damp*torch.mean(torch.diag(H))*torch.eye(H.shape[0],device=H.device)
    Hinv=torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(H)),upper=True); return W,Hinv

def gptq(W,H,bits,g,actorder=True):
    R,C=W.shape
    W0=W.clone()
    if actorder:
        perm=torch.argsort(torch.diag(H),descending=True); inv=torch.argsort(perm); W0=W0[:,perm]; H=H[perm][:,perm]
    W0,Hinv=prep_H(H,W0); W=W0
    Q=torch.zeros_like(W); codes=torch.zeros(R,C,device=W.device); meta=0
    for c in range(C):
        if c%g==0: s,z=qparams(W[:,c:c+g],bits); meta+=R*(16+bits)
        w=W[:,c]; q,d=qd(w[:,None],s,z,bits); Q[:,c]=d[:,0]; codes[:,c]=q[:,0]
        if c<C-1:
            err=(w-d[:,0])/Hinv[c,c]; W[:,c+1:]-=err[:,None]*Hinv[c,c+1:][None]
    if actorder: Q=Q[:,inv]; meta+=C*16
    return Q,codes,meta

def kmeans(X,w,k,iters=12):
    n=X.shape[0]; idx=torch.randperm(n,device=X.device)[:k]; C=X[idx].clone()
    for _ in range(iters):
        a=torch.cat([torch.argmin(torch.cdist(X[i:i+16384],C),1) for i in range(0,n,16384)])
        num=torch.zeros_like(C).index_add_(0,a,X*w[:,None]); den=torch.zeros(k,device=X.device).index_add_(0,a,w)
        e=den>0; C[e]=num[e]/den[e][:,None]
        if (~e).any(): C[~e]=X[torch.randint(0,n,(int((~e).sum()),),device=X.device)]
    return C
def vq_gptq(W,H,d,kb,g,actorder=True,cb=None):
    # GPTVQ-lite: groups of d input columns share a 2^kb codebook (fp16) after per-(row,g) rms scaling
    R,C=W.shape
    if actorder:
        perm=torch.argsort(torch.diag(H),descending=True); inv=torch.argsort(perm); W=W[:,perm]; H=H[perm][:,perm]
    W,Hinv=prep_H(H,W)
    hd=torch.diag(H).clamp(min=1e-8)
    sc=W.reshape(R,C//g,g).pow(2).mean(2).sqrt().half().float().clamp(min=1e-6)  # R,C/g
    scf=sc.repeat_interleave(g,1)
    Xn=(W/scf).reshape(R,C//d,d)
    wv=hd.reshape(C//d,d).mean(1)[None].expand(R,-1).reshape(-1)
    shared=cb is not None
    if not shared: Xv=Xn.reshape(-1,d); cb=kmeans(Xv,wv/wv.mean(),2**kb).half().float()
    Q=torch.zeros_like(W); codes=torch.zeros(R,C//d,device=W.device)
    for b in range(C//d):
        c=b*d; blk=W[:,c:c+d]/scf[:,c:c+d]
        a=torch.argmin(torch.cdist(blk,cb),1); codes[:,b]=a; deq=cb[a]*scf[:,c:c+d]; Q[:,c:c+d]=deq
        for j in range(d):
            cc=c+j
            if cc<C-1:
                err=(W[:,cc]-deq[:,j])/Hinv[cc,cc]; W[:,cc+1:]-=err[:,None]*Hinv[cc,cc+1:][None]
    meta=sc.numel()*16+(0 if shared else cb.numel()*16)+(C*16 if actorder else 0)
    if actorder: Q=Q[:,inv]
    return Q,codes,meta,2**kb

def quant_module(W,H,mc):
    """mc: bits,g,method(gptq|rtn|vq),awq(list of alphas),rank,outk,vq=(d,kb). returns Wq,(code_bits_huff,code_bits_fixed,code_bits_ent,meta_bits)"""
    R,C=W.shape; W=W.float(); meta=0
    hd=torch.diag(H)
    act=torch.ones(C,dtype=torch.bool,device=W.device)
    k=mc.get("outk",0)
    if k:
        sal=hd*W.pow(2).sum(0); top=torch.topk(sal,k)[1]; act[top]=False; meta+=k*R*16+k*16
    ai=act.nonzero()[:,0]; Wa=W[:,ai]; Ha=H[ai][:,ai]
    Lm=torch.zeros_like(Wa)
    if mc.get("rank",0):
        r=mc["rank"]; U,S,Vh=torch.linalg.svd(Wa,full_matrices=False)
        A=(U[:,:r]*S[:r]).half().float(); B=Vh[:r].half().float(); Lm=A@B; meta+=r*(R+Wa.shape[1])*16
    res=Wa-Lm; g=mc["g"]; bits=mc["bits"]; method=mc["method"]
    best=None
    for al in (mc.get("awq") or [0.0]):
        if al>0:
            s=torch.diag(Ha).clamp(min=1e-8).sqrt()**al; s=s/s.mean()
        else: s=torch.ones(Wa.shape[1],device=W.device)
        Ws=res*s; Hs=Ha/(s[:,None]*s[None])
        if method=="rtn": Q,codes,m_=rtn(Ws,bits,g); ns=2**bits
        elif method=="gptq": Q,codes,m_=gptq(Ws,Hs,bits,g,mc.get("actorder",True)); ns=2**bits
        else:
            d=mc["vq"][0]; kb=bits; Q,codes,m_,ns=vq_gptq(Ws,Hs,d,kb,g,mc.get("actorder",True),mc.get("cb"))
        Qe=Q/s; E=res-Qe; loss=float(((E@Ha)*E).sum())
        if best is None or loss<best[0]: best=(loss,Qe,codes,m_+(Wa.shape[1]*16 if al>0 else 0),ns)
    _,Qe,codes,m_,ns=best; meta+=m_
    Wq=W.clone(); Wq[:,ai]=Qe+Lm
    if method=="vq": fixed=codes.numel()*bits
    else: fixed=codes.numel()*bits
    return Wq,(huff_bits(codes,ns),fixed,ent_bits(codes,ns),meta)

def run(cfg, D2, D2H, vocab):
    import copy
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    torch.manual_seed(0)
    tok=AutoTokenizer.from_pretrained(M); m=AutoModelForSequenceClassification.from_pretrained(M).eval().to(D)
    T2=[x["text"] for x in D2]; y2=torch.tensor([x["label"] for x in D2]); TH=[x["text"] for x in D2H]
    cal=T2[:512]; ev=T2[512:]; y2=y2[512:]
    def bt(texts,bs=128):
        for i in range(0,len(texts),bs): yield {k:v.to(D) for k,v in tok(texts[i:i+bs],padding=True,truncation=True,max_length=64,return_tensors="pt").items()}
    @torch.no_grad()
    def preds(texts): return torch.cat([m(**b).logits.argmax(-1).cpu() for b in bt(texts)])
    @torch.no_grad()
    def logits(texts): return torch.cat([m(**b).logits.float().cpu() for b in bt(texts)])
    base_ev=preds(ev); base_h=preds(TH); base_cal=logits(cal)
    mods=[]
    for i,l in enumerate(m.bert.encoder.layer):
        mods+= [(f"{i}.q","att",l.attention.self.query),(f"{i}.k","att",l.attention.self.key),(f"{i}.v","att",l.attention.self.value),
                (f"{i}.o","att",l.attention.output.dense),(f"{i}.f1","f1",l.intermediate.dense),(f"{i}.f2","f2",l.output.dense)]
    mods.append(("pool","pool",m.bert.pooler.dense))
    # per-module bits
    bitsmap={n:cfg["bits"] for n,_,_ in mods}
    for n,t,_ in mods:
        if t in cfg.get("bytype",{}): bitsmap[n]=cfg["bytype"][t]
    if cfg.get("mix"):
        sens={}
        orig={n:mod.weight.data.clone() for n,_,mod in mods}
        for n,t,mod in mods:
            mod.weight.data=rtn(orig[n],2,64)[0]; sens[n]=float((logits(cal)-base_cal).pow(2).mean()); mod.weight.data=orig[n]
        order=sorted([n for n,_,_ in mods],key=lambda n:-sens[n]); f=cfg["mix"]  # list of (frac,bits) from most sensitive
        pos=0
        for frac,b in f:
            cnt=int(round(frac*len(order)))
            for n in order[pos:pos+cnt]: bitsmap[n]=b
            pos+=cnt
        for n in order[pos:]: bitsmap[n]=f[-1][1]
    tot=dict(huff=0,fixed=0,ent=0,meta=0)
    cbs={}
    if cfg["method"]=="vq" and cfg.get("vqshare"):
        d=cfg["vq"][0]; g=cfg["g"]
        for t in ("att","f1","f2"):
            ws=[mod.weight.data for n,tt,mod in mods if tt==t]
            Xs=[]
            for w in ws:
                R,C=w.shape; sc=w.reshape(R,C//g,g).pow(2).mean(2).sqrt().half().float().clamp(min=1e-6)
                Xs.append((w.reshape(R,C//g,g)/sc[:,:,None]).reshape(-1,d))
            X=torch.cat(Xs); X=X[torch.randperm(X.shape[0],device=X.device)[:cfg.get("vqsamp",1500000)]]
            for kb in sorted(set(bitsmap.values())):
                cbs[(t,kb)]=kmeans(X,torch.ones(X.shape[0],device=X.device),2**kb,iters=20).half().float()
                tot["meta"]+=cbs[(t,kb)].numel()*16
    def acc(i):
        tot["huff"]+=i[0];tot["fixed"]+=i[1];tot["ent"]+=i[2];tot["meta"]+=i[3]
    # sequential
    stages=[]
    for i in range(12): stages+= [mods[6*i:6*i+3],mods[6*i+3:6*i+4],mods[6*i+4:6*i+5],mods[6*i+5:6*i+6]]
    stages.append(mods[72:73])
    cur={}
    for st in stages:
        Hs={n:torch.zeros(mod.in_features,mod.in_features,device=D) for n,_,mod in st}
        if cfg["method"]!="rtn":
            def mk(n):
                def h(mod,inp,out):
                    x=inp[0].detach().float(); x=x.reshape(-1,x.shape[-1])
                    if cur["mask"].numel()==x.shape[0]: x=x[cur["mask"]]
                    Hs[n].add_(x.T@x)
                return h
            hk=[mod.register_forward_hook(mk(n)) for n,_,mod in st]
            with torch.no_grad():
                for b in bt(cal,64): cur["mask"]=b["attention_mask"].reshape(-1).bool(); m(**b)
            for h in hk: h.remove()
        for n,t,mod in st:
            mc=dict(method=cfg["method"],bits=bitsmap[n],g=cfg["g"],awq=cfg.get("awq"),rank=cfg.get("rank",0) if t!="pool" else 0,
                    outk=cfg.get("outk",0) if t!="pool" else 0,vq=cfg.get("vq"),actorder=cfg.get("actorder",True))
            if cfg.get("rank_ffn_only") and t not in("f1","f2"): mc["rank"]=0
            if cfg["method"]=="vq" and t=="pool": mc["method"]="gptq"; mc["bits"]=4
            if cfg["method"]=="vq" and cbs and t!="pool": mc["cb"]=cbs[(t if t!="att" else "att",bitsmap[n])]
            if mc["method"]=="rtn": Wq,info=quant_module(mod.weight.data,torch.eye(mod.in_features,device=D),mc)
            else: Wq,info=quant_module(mod.weight.data,Hs[n],mc)
            mod.weight.data=Wq; acc(info)
    # embeddings: vocab trim to ids, rest -> UNK
    ids=torch.tensor(vocab,device=D); emb=m.bert.embeddings.word_embeddings.weight.data
    eb=cfg.get("ebits",4); eg=cfg.get("eg",64); er=cfg.get("erank",0); E=emb[ids].clone()
    ebits_total=0
    if er:
        U,S,Vh=torch.linalg.svd(E,full_matrices=False); A=U[:,:er]*S[:er]; B=Vh[:er].half().float()
        Aq,codes,mt=rtn(A,eb,er if er<=64 else 64) if er%(64 if er>64 else er)==0 else (A,None,0)
        E=Aq@B; ebits_total=codes.numel()*eb+mt+er*768*16; ehuff=huff_bits(codes,2**eb)+mt+er*768*16
    else:
        E,codes,mt=rtn(E,eb,eg) if eb<16 else (E.half().float(),None,0)
        ebits_total=E.numel()*(eb if eb<16 else 16)+mt; ehuff=(huff_bits(codes,2**eb)+mt) if codes is not None else ebits_total
    new=emb[100:101].clone().repeat(emb.shape[0],1)  # unseen -> UNK row
    new[ids]=E; unk=(ids==100).nonzero()
    new[~torch.isin(torch.arange(emb.shape[0],device=D),ids)]=E[unk[0,0]] if len(unk) else 0
    m.bert.embeddings.word_embeddings.weight.data=new
    pos=cfg.get("pos",64); rest=pos*768
    for n,p in m.named_parameters():
        if "position_embeddings" in n: p.data[:pos]=p.data[:pos].half().float(); continue
        if "word_embeddings" in n or n.endswith("weight") and p.dim()==2 and any(k in n for k in ["query","key","value","attention.output.dense","intermediate.dense","output.dense","pooler"]): continue
        rest+=p.numel(); p.data=p.data.half().float()
    base=rest*16+tot["meta"]
    by_h=(tot["huff"]+ehuff+base)/8; by_f=(tot["fixed"]+ebits_total+base)/8
    pe=preds(ev); ph=preds(TH)
    return dict(cfg=cfg,bytes_huff=by_h,bytes_fixed=by_f,bytes_ent=(tot["ent"]+ebits_total+base)/8,acc2=float((pe==y2).float().mean()),
                base_acc2=float((base_ev==y2).float().mean()),agree2=float((pe==base_ev).float().mean()),agree_hard=float((ph==base_h).float().mean()),
                bitsmap_hist={b:sum(1 for v in bitsmap.values() if v==b) for b in set(bitsmap.values())})
