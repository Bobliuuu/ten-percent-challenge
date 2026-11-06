import modal, json, sys
app = modal.App("tenpct-vq")
img = (modal.Image.debian_slim().pip_install("torch","transformers==4.46.3","numpy","accelerate")
       .add_local_file("../base/dev.jsonl","/root/dev.jsonl").add_local_file("../base/calib.jsonl","/root/calib.jsonl"))
vol = modal.Volume.from_name("hf-cache", create_if_missing=True)
rvol = modal.Volume.from_name("vq-results", create_if_missing=True)

@app.function(image=img, gpu=["L40S","A100-40GB","A100-80GB","H100"], memory=49152, timeout=14400, max_containers=12, volumes={"/hf":vol,"/res":rvol})
def run(cfg):
    import os, zlib, time, math, re, torch, numpy as np, torch.nn as nn
    os.environ["HF_HOME"]="/hf"
    from transformers import AutoTokenizer, AutoModelForCausalLM
    torch.backends.cuda.matmul.allow_tf32=False
    t0=time.time(); dev="cuda"
    M=cfg.get("model","Qwen/Qwen2.5-14B-Instruct"); name=cfg["name"]
    tok=AutoTokenizer.from_pretrained(M)
    model=AutoModelForCausalLM.from_pretrained(M,torch_dtype=torch.bfloat16).eval()
    model.config.use_cache=False
    base_params=sum(p.numel() for p in model.parameters())
    rd=lambda f:[json.loads(l) for l in open(f)]
    ev=rd("/root/dev.jsonl"); calib=rd("/root/calib.jsonl")
    if cfg.get("nev"): ev=ev[:cfg["nev"]]
    def fmt(r,ans=False):
        s=f"The following are multiple choice questions (with answers) about {r['subject'].replace('_',' ')}.\n\n{r['question']}\n"
        for l,c in zip("ABCD",r["choices"]): s+=f"{l}. {c}\n"
        s+="Answer:"
        return s+(" "+"ABCD"[r["answer"]] if ans else "")
    letter_ids=[tok.encode(" "+l)[0] for l in "ABCD"]
    Wl=model.get_output_embeddings().weight.data[letter_ids].clone().cuda()
    y=np.array([r["answer"] for r in ev])
    @torch.no_grad()
    def predict(rows,bs=16):
        enc=[tok.encode(fmt(r))[-1024:] for r in rows]
        order=np.argsort([-len(e) for e in enc]); out=np.zeros((len(rows),4),dtype=np.float32)
        for b in range(0,len(rows),bs):
            idx=order[b:b+bs]; mx=max(len(enc[i]) for i in idx)
            ids=torch.zeros(len(idx),mx,dtype=torch.long); att=torch.zeros(len(idx),mx,dtype=torch.long)
            for j,i in enumerate(idx):
                ids[j,:len(enc[i])]=torch.tensor(enc[i]); att[j,:len(enc[i])]=1
            hs=model.model(input_ids=ids.cuda(),attention_mask=att.cuda()).last_hidden_state
            last=att.sum(1).cuda()-1
            out[idx]=(hs[torch.arange(len(idx)),last]@Wl.T).float().cpu().numpy()
        return out
    bkey=f"/res/baselg_{M.replace('/','_')}_{len(ev)}.npy"
    if os.path.exists(bkey): base_lg=np.load(bkey)
    else:
        model.cuda(); base_lg=predict(ev); model.cpu(); torch.cuda.empty_cache(); np.save(bkey,base_lg); rvol.commit()
    base_pred=base_lg.argmax(1); base_acc=float((base_pred==y).mean()); print("base_acc",base_acc,flush=True)

    # ---------- rotations
    def had(p):
        Hm=torch.ones(1,1)
        while Hm.shape[0]<p: Hm=torch.cat([torch.cat([Hm,Hm],1),torch.cat([Hm,-Hm],1)],0)
        return Hm/math.sqrt(p)
    class Rot:
        def __init__(s,n,seed):
            p=n&(-n); m=n//p; g=torch.Generator().manual_seed(seed)
            s.sg=(torch.randint(0,2,(n,),generator=g)*2-1).float().to(dev)
            Q=torch.linalg.qr(torch.randn(m,m,generator=g,dtype=torch.float64))[0] if m>1 else torch.ones(1,1,dtype=torch.float64)
            s.Q=Q.float().to(dev); s.H=had(p).to(dev); s.m=m; s.p=p
        def fwd(s,X):  # rotate last dim
            sh=X.shape; Y=(X*s.sg).reshape(-1,s.m,s.p)
            Y=torch.matmul(s.Q,Y); Y=torch.matmul(Y,s.H); return Y.reshape(sh)
        def inv(s,X):
            sh=X.shape; Y=X.reshape(-1,s.m,s.p)
            Y=torch.matmul(s.Q.T,Y); Y=torch.matmul(Y,s.H); return Y.reshape(sh)*s.sg
    # ---------- codebook (gaussian k-means, shared by all layers)
    D=8
    def get_cb(logk):
        f=f"/res/cb_{logk}.pt"
        if os.path.exists(f): return torch.load(f).to(dev)
        K=2**logk; g=torch.Generator(device=dev).manual_seed(logk)
        N=K*100; X=torch.randn(N,D,generator=g,device=dev); C=X[:K].clone()
        for it in range(25):
            sums=torch.zeros(K,D,device=dev); cnt=torch.zeros(K,device=dev); Cn=0.5*(C**2).sum(1)
            for a in range(0,N,16384):
                xb=X[a:a+16384]; ii=(xb@C.T-Cn).argmax(1)
                sums.index_add_(0,ii,xb); cnt.index_add_(0,ii,torch.ones(len(ii),device=dev))
            e=cnt==0; newC=sums/cnt.clamp(min=1).unsqueeze(1)
            newC[e]=X[torch.randint(0,N,(int(e.sum()),),device=dev,generator=g)]
            C=newC
        torch.save(C.cpu(),f+".tmp"); os.replace(f+".tmp",f); rvol.commit()
        return C
    cbs={}
    def nn_idx(X,C,Cn):
        out=torch.empty(X.shape[0],dtype=torch.int64,device=dev)
        torch.backends.cuda.matmul.allow_tf32=True
        for a in range(0,X.shape[0],4096):
            out[a:a+4096]=(X[a:a+4096]@C.T-Cn).argmax(1)
        torch.backends.cuda.matmul.allow_tf32=False
        return out
    def ldlq(Wn,H,C,chunk=128,damp=0.01):
        R,Cc=Wn.shape; Cn=0.5*(C**2).sum(1)
        H=(H+H.T)/2; dg=torch.diag(H); dead=dg==0; H[dead,dead]=1
        H=H.double(); md=torch.mean(torch.diag(H)); eye=torch.eye(Cc,device=dev,dtype=torch.float64)
        for dm in [damp,damp*10,damp*100,1.0]:
            try:
                Hd=H+dm*md*eye; U=torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(Hd)),upper=True).float(); break
            except Exception as ex: print("chol retry",dm,flush=True)
        del H,Hd
        W=Wn.clone(); W[:,dead]=0; idx=torch.empty(R,Cc//D,dtype=torch.int64,device=dev)
        for c0 in range(0,Cc,chunk):
            c1=min(c0+chunk,Cc); Wc=W[:,c0:c1].clone(); Err=torch.zeros_like(Wc)
            for j in range(0,c1-c0,D):
                blk=Wc[:,j:j+D]; i=nn_idx(blk,C,Cn); E=blk-C[i]
                Ub=U[c0+j:c0+j+D,c0+j:c0+j+D]
                Es=torch.linalg.solve_triangular(Ub,E,upper=True,left=False)
                Err[:,j:j+D]=Es
                if j+D<c1-c0: Wc[:,j+D:]-=Es@U[c0+j:c0+j+D,c0+j+D:c1]
                idx[:,(c0+j)//D]=i
            if c1<Cc: W[:,c1:]-=Err@U[c0:c1,c1:]
        return idx
    def decode(idx,C,sigma,rin,rout):  # exact decoder: idx -> effective weight (out,in) fp32
        Wq=(C[idx].reshape(idx.shape[0],-1))*sigma    # rotated frame
        Wq=rout.inv(Wq.T.contiguous()).T.contiguous()  # undo output rot
        return rin.inv(Wq)                             # undo input rot
    def pack(idx,logk):  # bit-pack to bytes (numpy) and unpack check
        a=idx.cpu().numpy().astype(np.uint32).ravel()
        bits=((a[:,None]>>np.arange(logk-1,-1,-1,dtype=np.uint32))&1).astype(np.uint8).ravel()
        return np.packbits(bits)
    def unpack(buf,n,logk):
        bits=np.unpackbits(buf)[:n*logk].reshape(n,logk).astype(np.uint32)
        return (bits<<np.arange(logk-1,-1,-1,dtype=np.uint32)).sum(1)
    # ---------- calibration stream
    rng=np.random.RandomState(0); txt=[fmt(r,True) for r in calib]; rng.shuffle(txt)
    ncal=cfg.get("ncal",256); L=cfg.get("callen",512)
    stream=[]
    for s in txt: stream+=tok.encode(s+"\n\n")
    stream=torch.tensor(stream[:ncal*L]).view(-1,L)[:ncal]
    layers=model.model.layers; nl0=len(layers)
    ins=[];kw={}
    class Catcher(nn.Module):
        def __init__(s,m): super().__init__(); s.m=m
        def forward(s,x,*a,**k): ins.append(x); kw.update(k); raise ValueError
    layers[0]=Catcher(layers[0])
    model.model.embed_tokens.cuda(); model.model.rotary_emb.cuda(); layers[0].m.cuda()
    for i in range(len(stream)):
        try: model(stream[i:i+1].cuda())
        except ValueError: pass
    layers[0]=layers[0].m
    cur=[x.clone() for x in ins]
    # ---------- layer drop
    ndrop=cfg.get("drop",0); dropl=[]
    if ndrop:
        bi=torch.zeros(nl0); c2=[x.clone() for x in cur]
        with torch.no_grad():
            for li,l in enumerate(layers):
                l.cuda(); s=0
                for j,x in enumerate(c2):
                    o=l(x,**kw); o=o[0] if isinstance(o,tuple) else o
                    s+=1-torch.nn.functional.cosine_similarity(x.float().flatten(0,1),o.float().flatten(0,1),dim=-1).mean().item(); c2[j]=o
                bi[li]=s/len(c2); l.cpu()
        del c2
        cand=list(range(2,nl0-1)); dropl=sorted(sorted(cand,key=lambda i:bi[i].item())[:ndrop])
        keepl=[i for i in range(nl0) if i not in dropl]
        model.model.layers=nn.ModuleList([layers[i] for i in keepl]); layers=model.model.layers
        for k,l in enumerate(layers): l.self_attn.layer_idx=k
        model.config.num_hidden_layers=len(layers); print("dropped",dropl,flush=True)
    orig_idx=[i for i in range(nl0) if i not in dropl]
    # ---------- quantize
    logk0=cfg.get("logk",14); plan=cfg.get("plan",[])  # list of [regex, logk]
    smult=cfg.get("smult",1.0)
    def lk(li,n):
        k=logk0
        for rg,v in plan:
            if re.search(rg,f"L{orig_idx[li]}.{n}"): k=v
        return k
    groups=[("q_proj",["self_attn.q_proj","self_attn.k_proj","self_attn.v_proj"],"self_attn.q_proj"),
            ("o_proj",["self_attn.o_proj"],"self_attn.o_proj"),
            ("gate_proj",["mlp.gate_proj","mlp.up_proj"],"mlp.gate_proj"),
            ("down_proj",["mlp.down_proj"],"mlp.down_proj")]
    st=dict(idx_bits=0.0,ent_bits=0.0,nq=0,meta_bytes=0,err_num=0.0,err_den=0.0,per={})
    verified=False
    for li,l in enumerate(layers):
        l.cuda(); mods={}; Hs={}; nsm={}
        for gname,names,hookn in groups:
            m=l.get_submodule(hookn); Hs[gname]=torch.zeros(m.in_features,m.in_features,device=dev); nsm[gname]=0
        def mk(gname):
            def hook(m,i,o):
                x=i[0].reshape(-1,i[0].shape[-1]).float(); nsm[gname]+=x.shape[0]; Hs[gname]+=x.T@x
            return hook
        hds=[l.get_submodule(hn).register_forward_hook(mk(gn)) for gn,_,hn in groups]
        with torch.no_grad():
            for x in cur: l(x,**kw)
        for h_ in hds: h_.remove()
        for gname,names,hookn in groups:
            Hraw=Hs[gname]/nsm[gname]
            n_in=Hraw.shape[0]; rin=Rot(n_in,zlib.crc32(f'{orig_idx[li]}.{gname}.in'.encode()))
            Hr=rin.fwd(Hraw); Hr=rin.fwd(Hr.T.contiguous()).T.contiguous()   # R H R^T
            for n in names:
                m=l.get_submodule(n); W=m.weight.data.float(); out_dim=W.shape[0]
                rout=Rot(out_dim,zlib.crc32(f'{orig_idx[li]}.{n}.out'.encode()))
                Wr=rin.fwd(W); Wr=rout.fwd(Wr.T.contiguous()).T.contiguous()
                sigma=float(Wr.std())*smult
                logk=lk(li,n)
                if logk not in cbs: cbs[logk]=get_cb(logk)
                C=cbs[logk]
                idx=ldlq(Wr/sigma,Hr,C)
                Q=(C[idx].reshape(out_dim,-1))*sigma
                Eq=(Wr-Q); num=float(((Eq@Hr)*Eq).sum()); den=float(((Wr@Hr)*Wr).sum())
                What=decode(idx,C,sigma,rin,rout)
                if not verified:
                    buf=pack(idx,logk); idx2=unpack(buf,idx.numel(),logk)
                    assert (idx2==idx.cpu().numpy().ravel()).all(), "pack roundtrip"
                    assert torch.equal(What,decode(idx,C,sigma,rin,rout)); verified=True
                    Qr=rout.fwd(rin.fwd(What).T.contiguous()).T.contiguous()
                    print("decode verify: rotated-frame maxdiff",float((Qr-Q).abs().max()),"packed bytes",len(buf),flush=True)
                m.weight.data=What.to(torch.bfloat16)
                nb=idx.numel(); cnt=torch.bincount(idx.flatten(),minlength=2**logk).float(); p=cnt/cnt.sum(); p=p[p>0]
                ent=float(-(p*p.log2()).sum())
                st["idx_bits"]+=nb*logk; st["ent_bits"]+=nb*ent; st["nq"]+=W.numel(); st["meta_bytes"]+=4+8
                st["err_num"]+=num; st["err_den"]+=den
                st["per"][f"L{orig_idx[li]}.{n}"]=dict(logk=logk,rel_err=num/den,ent=ent)
        del Hs
        with torch.no_grad():
            for j,x in enumerate(cur):
                o=l(x,**kw); cur[j]=o[0] if isinstance(o,tuple) else o
        l.cpu(); torch.cuda.empty_cache()
        print("layer",li,"relerr_so_far",st["err_num"]/st["err_den"],round(time.time()-t0),flush=True)
    # ---------- eval
    model.cuda(); lg=predict(ev); pred=lg.argmax(1); acc=float((pred==y).mean()); agree=float((pred==base_pred).mean())
    # ---------- bytes
    nbody=st["nq"]; other=sum(p.numel() for ly in layers for p_n,p in ly.named_parameters())-nbody
    other_b=other*2+model.model.norm.weight.numel()*2
    hd=model.config.hidden_size; V=model.config.vocab_size
    emb_trim=cfg.get("emb_bytes_est",32768*hd*0.5*(1+2/64*0+0.25))  # est trimmed 4-bit emb g64
    body_b=st["idx_bits"]/8+st["meta_bytes"]+sum(2**k*D*2 for k in cbs)
    body_e=st["ent_bits"]/8+st["meta_bytes"]+sum(2**k*D*2 for k in cbs)
    fixed=other_b+emb_trim+4*hd*2+1.0e6
    tot=body_b+fixed; tote=body_e+fixed
    res=dict(name=name,cfg=cfg,n=len(ev),base_acc=base_acc,acc=acc,agree=agree,
        flips_lost=int(((base_pred==y)&(pred!=y)).sum()),flips_gain=int(((base_pred!=y)&(pred==y)).sum()),
        dropped=dropl,body_bits_per_w=st["idx_bits"]/nbody,body_ent_bits_per_w=st["ent_bits"]/nbody,
        rel_proxy_err=st["err_num"]/st["err_den"],body_bytes=body_b,fixed_bytes_est=fixed,
        total_bytes_est=tot,ratio_bf16=tot/(base_params*2),ratio_bf16_entropy=tote/(base_params*2),
        budget=2954006733,secs=time.time()-t0,per=st["per"] if cfg.get("keep_per") else None)
    json.dump(res,open(f"/res/{name}.json","w"),default=str); rvol.commit()
    print("RESULT_JSON "+json.dumps(res,default=str),flush=True); return res

@app.local_entrypoint()
def main(cfgs:str):
    cs=json.load(open(cfgs))
    for r in run.map(cs,return_exceptions=True):
        print(json.dumps(r,default=str)[:400])
