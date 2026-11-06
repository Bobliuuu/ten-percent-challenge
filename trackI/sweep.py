import modal, json, sys
app = modal.App("tenpct-trackI")
img = (modal.Image.debian_slim().pip_install("torch","transformers","datasets","numpy","sentencepiece","protobuf","tiktoken")
       .add_local_file("dev2.json","/root/dev2.json").add_local_file("dev2_hard.json","/root/dev2_hard.json").add_local_file("embq.py","/root/embq.py"))

@app.function(image=img, gpu="A100", timeout=3600, max_containers=20)
def run(M):
    import torch, numpy as np, time, copy, traceback, sys
    from datasets import load_dataset
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    t0=time.time(); sys.path.insert(0,'/root'); import embq, collections
    try:
        tok=AutoTokenizer.from_pretrained(M)
        m0=AutoModelForSequenceClassification.from_pretrained(M,torch_dtype=torch.float32).cuda().eval()
    except Exception as e:
        return dict(model=M,error=repr(e)[:300])
    D=lambda f:[(x["text"],x["label"]) for x in json.load(open(f))]
    d2=D("/root/dev2.json"); dh=D("/root/dev2_hard.json")
    id2=m0.config.id2label; print(M,id2)
    def preds(model,data):
        out=[]
        with torch.no_grad():
            for i in range(0,len(data),64):
                b=tok([t for t,_ in data[i:i+64]],padding=True,truncation=True,max_length=128,return_tensors="pt").to("cuda")
                out.append(model(**b).logits.float().cpu())
        return torch.cat(out)
    def tobin(lg):
        # map to 0/1: 2-class direct; 3-class use neg(0) vs pos(2)
        if lg.shape[1]==3: lg=lg[:,[0,2]]
        return lg.argmax(-1).numpy()
    # label polarity check: id2label names
    names=[str(v).lower() for v in id2.values()]
    flip = names[:2]==["positive","negative"] or (len(names)==2 and names[0].startswith("pos"))
    y2=np.array([l for _,l in d2]); 
    B2=preds(m0,d2); b2=tobin(B2); 
    if flip: b2=1-b2
    Bh=preds(m0,dh); bh=tobin(Bh)
    if flip: bh=1-bh
    nparam=sum(p.numel() for p in m0.parameters()); base_bytes=nparam*4
    emb=m0.get_input_embeddings().weight; V,H=emb.shape
    out=dict(model=M,params=nparam,base_mb=base_bytes/1e6,vocab=V,hidden=H,emb_share=V*H/nparam,flip=flip,
             base_acc_dev2=float((b2==y2).mean()),labels=names,res=[])
    # vocab trim: tokens in SST-2 TRAIN
    tr=list(load_dataset("nyu-mll/glue","sst2")["train"]["sentence"][:])
    ids=set(tok.all_special_ids)
    enc=tok(tr,truncation=True,max_length=128)["input_ids"]
    for e in enc: ids.update(e)
    ids=sorted(i for i in ids if i<V); out["trim_vocab"]=len(ids)
    cnt=collections.Counter()
    for e in enc: cnt.update(e)
    freq=torch.zeros(V)
    for k_,v_ in cnt.items():
        if k_<V: freq[k_]=v_
    for s_ in tok.all_special_ids:
        if s_<V: freq[s_]=1e9
    unk=tok.unk_token_id if tok.unk_token_id is not None else (ids[0])
    # linears to quantize: all nn.Linear in encoder body (names not containing classifier/pooler/score/lm)
    def qlinears(model):
        L=[]
        for n,mod in model.named_modules():
            if isinstance(mod,torch.nn.Linear) and not any(k in n for k in ("classifier","pooler","score","cls_head","out_proj_cls")) :
                # skip final head dense: classifier head in roberta = classifier.dense/out_proj
                L.append((n,mod))
        return L
    LIN=qlinears(m0)
    n_lin=sum(mod.weight.numel() for _,mod in LIN)
    out["lin_params"]=n_lin; out["lin_share"]=n_lin/nparam
    rest_params=nparam-n_lin-V*H  # fp16 rest
    # calibration: first 512 of dev2 ; eval: dev2[512:]
    calib=[t for t,_ in d2[:512]]
    # Hessians
    def collect_H():
        Hs={}; cnt={}
        hooks=[]
        for n,mod in qlinears(m0):
            Hs[n]=torch.zeros(mod.in_features,mod.in_features,device="cuda")
            cnt[n]=0
            def hk(mod_,inp,_,n=n):
                x=inp[0].reshape(-1,inp[0].shape[-1]).float()
                Hs[n].addmm_(x.T,x); cnt[n]+=x.shape[0]
            hooks.append(mod.register_forward_hook(hk))
        with torch.no_grad():
            for i in range(0,len(calib),32):
                b=tok(calib[i:i+32],padding=True,truncation=True,max_length=128,return_tensors="pt").to("cuda")
                m0(**b)
        for h in hooks: h.remove()
        return {n:Hs[n]*2/cnt[n] for n in Hs}
    Hs=collect_H()
    def qg(w,bits,g):
        sh=w.shape; w2=w.reshape(-1,g)
        lo=w2.min(1,keepdim=True)[0]; hi=w2.max(1,keepdim=True)[0]
        n=2**bits-1; s=(hi-lo).clamp(min=1e-8)/n
        return (((w2-lo)/s).round().clamp(0,n)*s+lo).reshape(sh)
    def gptq(W,H,bits,g,damp=0.01,blk=128):
        W=W.clone().float(); rows,cols=W.shape
        H=H.clone(); dead=torch.diag(H)==0; H[dead,dead]=1; W[:,dead]=0
        perm=torch.argsort(torch.diag(H),descending=True); W=W[:,perm]; H=H[perm][:,perm]
        H+=damp*torch.diag(H).mean()*torch.eye(cols,device=W.device)
        Hinv=torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(H)),upper=True)
        Q=torch.zeros_like(W); n=2**bits-1
        # static groups on permuted order (scale/zero per g consecutive permuted cols)
        for i1 in range(0,cols,blk):
            i2=min(i1+blk,cols); W1=W[:,i1:i2].clone(); Q1=torch.zeros_like(W1); E1=torch.zeros_like(W1); Hi=Hinv[i1:i2,i1:i2]
            for i in range(i2-i1):
                if (i1+i)%g==0:
                    blkw=W1[:,i:i+g] if i+g<=i2-i1 else torch.cat([W1[:,i:],W[:,i2:i1+i+g]],1)
                    lo=blkw.min(1)[0]; hi=blkw.max(1)[0]
                    s=((hi-lo).clamp(min=1e-8)/n); z=lo
                w=W1[:,i]; q=((w-z)/s).round().clamp(0,n)*s+z
                Q1[:,i]=q; e=(w-q)/Hi[i,i]; W1[:,i:]-=e[:,None]*Hi[i,i:][None]; E1[:,i]=e
            Q[:,i1:i2]=Q1; W[:,i2:]-=E1@Hinv[i1:i2,i2:]
        inv=torch.argsort(perm); return Q[:,inv]
    EMB_RARE='rtn2'
    def build(method,bits,g,ebits,trim=True):
        m=copy.deepcopy(m0)
        if bits<16:
            for n,mod in qlinears(m):
                w=mod.weight.data
                if w.shape[1]%g: continue
                mod.weight.data= qg(w,bits,g) if method=="rtn" else gptq(w,Hs[n],bits,g).to(w.dtype)
        E=m.get_input_embeddings().weight.data
        if trim:
            e,_=embq.quantize_embedding(E,freq,rare=EMB_RARE,hot_bits=ebits) if ebits<16 else (E,0)
            m.get_input_embeddings().weight.data=e
        return m
    def evalm(m):
        p2=tobin(preds(m,d2[512:])); ph=tobin(preds(m,dh))
        if flip: p2=1-p2; ph=1-ph
        return dict(agree=float((p2==b2[512:]).mean()),agree_hard=float((ph==bh).mean()),acc=float((p2==y2[512:]).mean()))
    EMB_BYTES={}
    def bytes_(bits,g,ebits,trim=True):
        nv=V
        b=n_lin*bits/8+(n_lin/g)*4 if bits<16 else n_lin*2
        if ebits<16:
            if ebits not in EMB_BYTES: EMB_BYTES[ebits]=embq.quantize_embedding(m0.get_input_embeddings().weight.data,freq,rare=EMB_RARE,hot_bits=ebits)[1]
            b+=EMB_BYTES[ebits]
        else: b+=nv*H*2
        b+=rest_params*2
        return b
    cfgs=[("fp",16,0,16,False),("fp",16,0,16,True),("fp",16,0,4,True)]
    for method in ("rtn","gptq"):
        for bits in (4,3,2):
            for g in (64,128):
                cfgs.append((method,bits,g,4,True))
    for (method,bits,g,eb,trim) in cfgs:
        try:
            m=build(method,bits,g,eb,trim)
            r=evalm(m); r.update(method=method,bits=bits,g=g,ebits=eb,trim=trim,ratio=bytes_(bits,g or 64,eb,trim)/base_bytes)
            out["res"].append(r); del m; torch.cuda.empty_cache()
            print(M,r,flush=True)
        except Exception as e:
            out["res"].append(dict(method=method,bits=bits,g=g,error=repr(e)[:200]))
    out["secs"]=time.time()-t0
    return out

@app.local_entrypoint()
def main(models:str=""):
    ms=models.split(",")
    res=list(run.map(ms))
    json.dump(res,open("results.json","w"),indent=1)
    for r in res: print(r.get("model"),r.get("error",""),r.get("base_acc_dev2"),r.get("emb_share"))
