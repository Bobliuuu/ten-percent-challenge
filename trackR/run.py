import modal, json, sys
app = modal.App("tenpct-trackR")
img = (modal.Image.debian_slim().pip_install("torch","transformers==4.46.3","datasets","numpy","accelerate")
       .add_local_file("split.py","/root/split.py"))
vol = modal.Volume.from_name("hf-cache", create_if_missing=True)

@app.function(image=img, gpu="A100-80GB", timeout=7200, max_containers=40, volumes={"/hf":vol},
              secrets=[])
def run(cfg):
    import os, sys, time, math, torch, numpy as np, torch.nn as nn
    os.environ["HF_HOME"]="/hf"; sys.path.insert(0,"/root")
    from datasets import load_dataset
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from split import h
    t0=time.time()
    M=cfg["model"]; split=cfg.get("split","dev")
    tok=AutoTokenizer.from_pretrained(M)
    model=AutoModelForCausalLM.from_pretrained(M,torch_dtype=torch.bfloat16).cuda().eval()
    model.config.use_cache=False
    base_params=sum(p.numel() for p in model.parameters())  # tied counted once
    task=cfg.get("task","mmlu")
    if task=="mmlu":
        ds=load_dataset("cais/mmlu","all"); T=ds["test"]
        calsrc=list(ds["validation"])+list(ds["dev"]); auxsrc=ds["auxiliary_train"]
    else:
        ag=load_dataset("fancyzhx/ag_news")
        mk=lambda r:dict(subject="news topic classification",question=r["text"],choices=["World","Sports","Business","Science and Technology"],answer=r["label"])
        T=[mk(r) for r in ag["test"]]
        rr=np.random.RandomState(1).permutation(len(ag["train"]))[:3000]
        calsrc=[mk(ag["train"][int(i)]) for i in rr]; auxsrc=None
    def fmt(r,ans=False):
        s=f"The following are multiple choice questions (with answers) about {r['subject'].replace('_',' ')}.\n\n" if task=='mmlu' else 'Classify the topic of the news article.\n\nArticle: '
        s+=f"{r['question']}\n"
        for l,c in zip("ABCD",r["choices"]): s+=f"{l}. {c}\n"
        s+="Answer:"
        return s+(" "+"ABCD"[r["answer"]] if ans else "")
    want=0 if split=="locked" else 1
    ev=[r for r in T if h(r)%7==want]
    other=[r for r in T if h(r)%7 not in (0,)]  # non-locked test text allowed for vocab only
    letter_ids=[tok.encode(" "+l)[0] for l in "ABCD"]
    # ---- calibration data (never locked)
    rng=np.random.RandomState(0)
    cal_txt=[fmt(r,True) for r in calsrc]
    ai=rng.permutation(len(auxsrc))[:600] if auxsrc is not None else []
    for i in ai:
        r=auxsrc[int(i)]; cal_txt.append(r["question"]+"\n"+"\n".join(f"{l}. {c}" for l,c in zip("ABCD",r["choices"]))+"\nAnswer: "+"ABCD"[r["answer"]])
    rng.shuffle(cal_txt)
    ncal=cfg.get("ncal",128); L=cfg.get("callen",512)
    # pack to ncal sequences of L tokens
    stream=[]
    for s in cal_txt: stream+=tok.encode(s+"\n\n")
    stream=torch.tensor(stream[:ncal*L]).view(-1,L)[:ncal]
    # ---- vocab trimming (keep set from non-locked text only)
    V=model.get_input_embeddings().weight.shape[0]
    keepN=cfg.get("vocab",0)
    cnt=torch.zeros(V)
    if keepN:
        for r in other[:6000]+calsrc: 
            ids=tok.encode(fmt(r,True)); cnt.index_add_(0,torch.tensor(ids),torch.ones(len(ids)))
        keep=set(cnt.topk(keepN)[1].tolist())
        for i in range(V):  # all single-char (byte) tokens
            pass
        vocab_inv=tok.convert_ids_to_tokens(list(range(len(tok))))
        tok2id=tok.get_vocab()
        for t_,i in tok2id.items():
            if len(t_)==1: keep.add(i)
        keep|=set(letter_ids)|set(tok.all_special_ids)
        keepmask=torch.zeros(V,dtype=torch.bool); keepmask[list(keep)]=True
        nkeep=int(keepmask.sum())
        keepstr={vocab_inv[i]:i for i in keep if i<len(vocab_inv)}
        maxl=max(len(k) for k in keepstr)
        oov_count=[0,0]
        def remap(ids):
            out=[]
            for i in ids:
                oov_count[1]+=1
                if i<V and keepmask[i]: out.append(i); continue
                oov_count[0]+=1
                t_=vocab_inv[i] if i<len(vocab_inv) else None
                if t_ is None: out.append(i); continue
                p=0
                while p<len(t_):
                    for q in range(min(len(t_),p+maxl),p,-1):
                        if t_[p:q] in keepstr: out.append(keepstr[t_[p:q]]); p=q; break
                    else: out.append(i); p+=1
            return out
    else:
        nkeep=V; remap=lambda ids: ids
    # ---- eval
    Wl=model.get_output_embeddings().weight.data[letter_ids].clone()  # 4 rows, fp16 stored
    @torch.no_grad()
    def predict(rows,bs=16):
        enc=[remap(tok.encode(fmt(r)))[-1024:] for r in rows]
        order=np.argsort([-len(e) for e in enc]); out=np.zeros((len(rows),4),dtype=np.float32)
        for b in range(0,len(rows),bs):
            idx=order[b:b+bs]; mx=max(len(enc[i]) for i in idx)
            ids=torch.zeros(len(idx),mx,dtype=torch.long); att=torch.zeros(len(idx),mx,dtype=torch.long)
            for j,i in enumerate(idx):
                ids[j,:len(enc[i])]=torch.tensor(enc[i]); att[j,:len(enc[i])]=1
            hs=model.model(input_ids=ids.cuda(),attention_mask=att.cuda()).last_hidden_state
            last=att.sum(1).cuda()-1
            hv=hs[torch.arange(len(idx)),last]
            out[idx]=(hv@Wl.T).float().cpu().numpy()
        return out
    y=np.array([r["answer"] for r in ev])
    def score(lg): return float((lg.argmax(1)==y).mean())
    # base predictions (bf16, full vocab, full model)
    global_remap=remap
    remap=lambda ids: ids
    base_lg=predict(ev); base_pred=base_lg.argmax(1); base_acc=score(base_lg)
    remap=global_remap
    # ---- layer drop (block influence on calibration)
    layers=model.model.layers
    nl0=len(layers)
    def get_inps(n):
        ins=[];kw={}
        class Catcher(nn.Module):
            def __init__(s,m): super().__init__(); s.m=m
            def forward(s,x,*a,**k): ins.append(x); kw.update(k); raise ValueError
        layers[0]=Catcher(layers[0])
        for i in range(n):
            try: model(stream[i:i+1].cuda())
            except ValueError: pass
        layers[0]=layers[0].m
        return ins,kw
    ndrop=cfg.get("drop",0)
    inps,kw=get_inps(len(stream))
    if ndrop:
        bi=torch.zeros(nl0)
        cur=[x.clone() for x in inps]
        with torch.no_grad():
            for li,l in enumerate(layers):
                s=0
                for j,x in enumerate(cur):
                    o=l(x,**kw); o=o[0] if isinstance(o,tuple) else o
                    s+=1-torch.nn.functional.cosine_similarity(x.float().flatten(0,1),o.float().flatten(0,1),dim=-1).mean().item()
                    cur[j]=o
                bi[li]=s/len(cur)
        # never drop first 2 / last 1
        cand=[i for i in range(2,nl0-1)]
        # drop n smallest (contiguous-agnostic)
        dropl=sorted(sorted(cand,key=lambda i:bi[i].item())[:ndrop])
        keepl=[i for i in range(nl0) if i not in dropl]
        model.model.layers=nn.ModuleList([layers[i] for i in keepl]); layers=model.model.layers
        for new_i,l in enumerate(layers): l.self_attn.layer_idx=new_i
        model.config.num_hidden_layers=len(layers)
        print("dropped",dropl,flush=True)
    else: dropl=[]
    # ---- quantizers
    bits=cfg.get("bits",16); g=cfg.get("g",128); method=cfg.get("method","gptq"); nm24=cfg.get("nm24",False)
    edge=cfg.get("edge_bits",None)
    stats={"nq":0,"bitsum":0.0,"ent_bitsum":0.0,"scale_bits":0.0,"nz":0}
    def qparams(x,b):  # x: rows x g -> scale,zero per row, with clip search
        n=2**b-1
        xmin=x.min(1,keepdim=True)[0].clamp(max=0)*0+x.min(1,keepdim=True)[0]; xmax=x.max(1,keepdim=True)[0]
        best=None
        for p in ([1.0,0.95,0.9,0.85,0.8,0.7,0.6,0.5] if b<=3 else [1.0,0.95,0.9,0.85]):
            lo=xmin*p; hi=xmax*p
            s=((hi-lo)/n).clamp(min=1e-8); z=torch.round(-lo/s)
            q=(torch.clamp(torch.round(x/s)+z,0,n)-z)*s
            e=((q-x)**2).sum(1,keepdim=True)
            if best is None: best=(e,s,z)
            else:
                m=e<best[0]; best=(torch.where(m,e,best[0]),torch.where(m,s,best[1]),torch.where(m,z,best[2]))
        return best[1],best[2]
    def quantize(x,s,z,b):
        n=2**b-1
        qi=torch.clamp(torch.round(x/s)+z,0,n)
        return (qi-z)*s,qi
    def rtn(W,b):
        R,C=W.shape; W=W.float(); Q=torch.zeros_like(W); hist=torch.zeros(2**b,device=W.device)
        for c in range(0,C,g):
            blk=W[:,c:c+g]; s,z=qparams(blk,b); q,qi=quantize(blk,s,z,b); Q[:,c:c+g]=q
            hist+=torch.bincount(qi.long().flatten(),minlength=2**b).float()
        return Q,hist
    def gptq(W,H,b,prune24=False,blocksize=128,damp=0.01):
        W=W.float().clone(); R,C=W.shape
        dead=torch.diag(H)==0; H[dead,dead]=1; W[:,dead]=0
        H=H+damp*torch.mean(torch.diag(H))*torch.eye(C,device=H.device)
        Hinv=torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(H)),upper=True)
        Q=torch.zeros_like(W); hist=torch.zeros(2**b,device=W.device)
        s=z=None
        for i1 in range(0,C,blocksize):
            i2=min(i1+blocksize,C); W1=W[:,i1:i2].clone(); Q1=torch.zeros_like(W1); Err1=torch.zeros_like(W1); Hi=Hinv[i1:i2,i1:i2]
            for i in range(i2-i1):
                w=W1[:,i]; d=Hi[i,i]
                if prune24 and i%4==0:
                    sal=W1[:,i:i+4]**2/torch.diag(Hi)[i:i+4].reshape(1,-1)**2
                    idx=torch.topk(sal,2,dim=1,largest=False)[1]
                    mask=torch.zeros_like(sal,dtype=torch.bool); mask.scatter_(1,idx,True); m4=mask
                if (i1+i)%g==0:
                    s,z=qparams(W[:,i1+i:i1+i+g] if i==0 and False else torch.cat([W1[:,i:],W[:,i2:]],1)[:,:g] if False else W1[:,i:i+g] if i+g<=i2-i1 else W1[:,i:],b)
                q,qi=quantize(w.unsqueeze(1),s,z,b); q=q.flatten()
                if prune24:
                    q=torch.where(m4[:,i%4],torch.zeros_like(q),q)
                else: hist+=torch.bincount(qi.long().flatten(),minlength=2**b).float()
                Q1[:,i]=q; err=(w-q)/d
                W1[:,i:]-=err.unsqueeze(1)@Hi[i,i:].unsqueeze(0); Err1[:,i]=err
            Q[:,i1:i2]=Q1; W[:,i2:]-=Err1@Hinv[i1:i2,i2:]
        return Q,hist
    def entropy(hist):
        p=hist/hist.sum(); p=p[p>0]; return float(-(p*p.log2()).sum())
    linnames=["self_attn.q_proj","self_attn.k_proj","self_attn.v_proj","self_attn.o_proj","mlp.gate_proj","mlp.up_proj","mlp.down_proj"]
    body_params=0
    if bits<16 or nm24:
        cur=[x.clone() for x in inps]  # inputs to (new) layer 0: recompute since layers changed? inputs to layer0 unchanged
        for li,l in enumerate(layers):
            b=edge if (edge and (li in (0,len(layers)-1))) else bits
            mods={n:l.get_submodule(n) for n in linnames}
            Hs={n:torch.zeros(m.in_features,m.in_features,device="cuda",dtype=torch.float32) for n,m in mods.items()}
            nsm={n:0 for n in mods}
            def mk(n):
                def hook(m,i,o):
                    x=i[0].reshape(-1,i[0].shape[-1]).float(); nsm[n]+=x.shape[0]; Hs[n]+=2*x.T@x
                return hook
            hds=[m.register_forward_hook(mk(n)) for n,m in mods.items()]
            with torch.no_grad():
                for x in cur: l(x,**kw)
            for hh in hds: hh.remove()
            for n,m in mods.items():
                H=Hs[n]/nsm[n]
                if method=="rtn" and not nm24: Q,hist=rtn(m.weight.data,b)
                else: Q,hist=gptq(m.weight.data,H,b,prune24=nm24)
                m.weight.data=Q.to(torch.bfloat16)
                np_=Q.numel(); body_params+=np_
                stats["nq"]+=np_; stats["bitsum"]+=np_*b; stats["scale_bits"]+=np_/g*(16+b)
                if not nm24: stats["ent_bitsum"]+=np_*entropy(hist)
                else: stats["nz"]+=int((Q==0).sum())
            del Hs
            with torch.no_grad():
                for j,x in enumerate(cur):
                    o=l(x,**kw); cur[j]=o[0] if isinstance(o,tuple) else o
            print("layer",li,time.time()-t0,flush=True)
    else:
        body_params=sum(l.get_submodule(n).weight.numel() for l in layers for n in linnames)
    # embedding quantization (trimmed rows only counted); simulate int4 g64 RTN on embedding if requested
    emb=model.get_input_embeddings(); hdim=emb.weight.shape[1]
    ebits=cfg.get("ebits",16)
    if ebits<16:
        Q,_=rtn(emb.weight.data,ebits) if False else (None,None)
        w=emb.weight.data.float(); gg=64; sh=w.shape; wq=torch.zeros_like(w)
        for c in range(0,sh[1],gg):
            blk=w[:,c:c+gg]; s,z=qparams(blk,ebits); wq[:,c:c+gg]=quantize(blk,s,z,ebits)[0]
        emb.weight.data=wq.to(torch.bfloat16)
        if model.config.tie_word_embeddings: pass  # letter rows Wl cloned earlier from original
    # ---- score
    lg=predict(ev); pred=lg.argmax(1); acc=score(lg)
    agree=float((pred==base_pred).mean())
    flips_b=int(((base_pred==y)&(pred!=y)).sum()); flips_c=int(((base_pred!=y)&(pred==y)).sum())
    # ---- byte accounting
    tied=model.config.tie_word_embeddings
    base_bf16=base_params*2; base_fp32=base_params*4
    total_orig_linear=sum(l.get_submodule(n).weight.numel() for l in model.model.layers for n in linnames)
    other_layer_params=sum(p.numel() for l in model.model.layers for n,p in l.named_parameters() if not any(n.startswith(x) and n.endswith("weight") for x in linnames))
    other_layer_params-=0
    other_layer_params=sum(p.numel() for l in model.model.layers for n,p in l.named_parameters())-total_orig_linear
    final_norm=model.model.norm.weight.numel()
    quant_bits_ent=(stats["ent_bitsum"] if stats["nq"] and not nm24 else stats["bitsum"])
    if bits>=16 and not nm24: stats["bitsum"]=total_orig_linear*16; quant_bits_ent=stats["bitsum"]
    if nm24:
        # 2:4 : nonzeros stored at b bits (half the weights) + mask 2.585 bits per 4 (log2 6)
        wbits=stats["nq"]/2*b+stats["nq"]/4*math.log2(6)
        wbits_ent=wbits
    else:
        wbits=stats["bitsum"]; wbits_ent=quant_bits_ent
    extra=(stats["scale_bits"] if bits<16 else 0)
    emb_bits=nkeep*hdim*(ebits if ebits<16 else 16)+(nkeep*hdim/64*(16+ebits) if ebits<16 else 0)
    head_bits=4*hdim*16 + (0 if tied else 0)
    fixed_bits=(other_layer_params+final_norm)*16 + 0.5e6*8   # biases+norms fp16, tokenizer/code slack 0.5MB
    if not keepN: # full vocab embedding stays; untied lm_head would also stay
        head_bits=(0 if tied else V*hdim*16)+4*hdim*16*0
    tot=wbits+extra+emb_bits+head_bits+fixed_bits
    tot_ent=wbits_ent+extra+emb_bits+head_bits+fixed_bits
    res=dict(cfg=cfg,n=len(ev),base_acc=base_acc,acc=acc,agree=agree,b_wrong=flips_b,c_fixed=flips_c,
        params=base_params,tied=bool(tied),nkeep=nkeep,vocab=V,dropped=dropl,
        bytes=tot/8,bytes_entropy=tot_ent/8,ratio_bf16=tot/8/base_bf16,ratio_fp32=tot/8/base_fp32,
        ratio_bf16_ent=tot_ent/8/base_bf16,ratio_fp32_ent=tot_ent/8/base_fp32,
        body_bits_per_w=(wbits+extra)/max(1,stats["nq"]) if stats["nq"] else 16,
        body_bits_per_w_ent=(wbits_ent+extra)/max(1,stats["nq"]) if stats["nq"] else 16,
        body_frac_params=total_orig_linear/base_params,oov=(oov_count[0]/max(1,oov_count[1]) if keepN else 0), secs=time.time()-t0)
    print(json.dumps(res),flush=True)
    return res

@app.local_entrypoint()
def main(cfgs:str, out:str):
    cs=json.load(open(cfgs))
    rs=[]
    for r in run.map(cs,return_exceptions=True):
        rs.append(r if not isinstance(r,Exception) else {"error":repr(r)})
        json.dump(rs,open(out,"w"),indent=1)
    for r in rs: print(json.dumps(r)[:600])
