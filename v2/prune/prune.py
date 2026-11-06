"""v2/prune: structured removal + closed-form recovery + GPTQ on Qwen2.5-14B-Instruct. NO gradient training.
Pipeline per cfg: base preds -> [layer tie] -> [contiguous block drop (angular distance) + closed-form (reduced-rank) LS patch]
 -> sequential per-layer [FFN neuron prune + LS down_proj reconstruction] [KV-group head prune + LS o_proj reconstruction]
 -> [GPTQ bits/g (stacked rows, entropy est)] [+ low-rank (H-weighted SVD) correction of quant error] -> dev eval -> exact byte accounting.
Launch: modal run --detach prune.py --cfgs cfgs.json     (driver is remote; results -> volume tenpct-prune + RESULT_JSON log lines)
"""
import modal, json
app = modal.App("tenpct-v2prune")
img = (modal.Image.debian_slim().pip_install("torch","transformers==4.46.3","numpy","accelerate")
       .add_local_file("../base/dev.jsonl","/root/base/dev.jsonl").add_local_file("../base/calib.jsonl","/root/base/calib.jsonl"))
vol = modal.Volume.from_name("hf-cache", create_if_missing=True)
ovol = modal.Volume.from_name("tenpct-prune", create_if_missing=True)
BASE_BYTES = 29540067328

@app.function(image=img, gpu=["H100","A100-80GB"], timeout=14400, max_containers=40, volumes={"/hf":vol,"/out":ovol})
def run(cfg):
    import os, sys, time, math, torch, numpy as np, torch.nn as nn, torch.nn.functional as F
    os.environ["HF_HOME"]="/hf"
    from transformers import AutoTokenizer, AutoModelForCausalLM
    t0=time.time(); name=cfg["name"]
    M="Qwen/Qwen2.5-14B-Instruct"
    tok=AutoTokenizer.from_pretrained(M)
    model=AutoModelForCausalLM.from_pretrained(M,torch_dtype=torch.bfloat16).cuda().eval()
    model.config.use_cache=False
    for p in model.parameters(): p.requires_grad_(False)
    D=model.config.hidden_size
    rd=lambda f:[json.loads(x) for x in open("/root/base/"+f)]
    DEV=rd("dev.jsonl"); CAL=rd("calib.jsonl")
    if cfg.get("nev"): DEV=DEV[:cfg["nev"]]
    def fmt(r,ans=False):
        s=f"The following are multiple choice questions (with answers) about {r['subject'].replace('_',' ')}.\n\n{r['question']}\n"
        for l,c in zip("ABCD",r["choices"]): s+=f"{l}. {c}\n"
        s+="Answer:"
        return s+(" "+"ABCD"[r["answer"]] if ans else "")
    letter_ids=[tok.encode(" "+l)[0] for l in "ABCD"]
    Wl=model.get_output_embeddings().weight.data[letter_ids].clone()
    y=np.array([r["answer"] for r in DEV])
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
    # ---- calibration stream
    rng=np.random.RandomState(0)
    cal_txt=[fmt(r,True) for r in CAL]; rng.shuffle(cal_txt)
    ncal=cfg.get("ncal",128); CL=cfg.get("callen",512)
    stream=[]
    for s in cal_txt: stream+=tok.encode(s+"\n\n")
    stream=torch.tensor(stream[:ncal*CL]).view(-1,CL)[:ncal]
    # base
    if cfg.get("mode")=="delta":
        res=delta_analysis(model)
        print("RESULT_JSON "+json.dumps(dict(name=name,cfg=cfg,**res)),flush=True)
        json.dump(dict(name=name,cfg=cfg,**res),open(f"/out/{name}.json","w")); ovol.commit(); return res
    base_lg=predict(DEV); base_pred=base_lg.argmax(1); base_acc=float((base_pred==y).mean())
    print("base_acc",base_acc,time.time()-t0,flush=True)
    layers=model.model.layers
    # ---- layer tie (shared weights, zero extra bytes)
    ties=cfg.get("tie",[])
    for src,dst in ties:
        layers[dst].load_state_dict(layers[src].state_dict())
    tied_layers=len(ties)
    # ---- layer drop + patch
    class Patch(nn.Module):
        def __init__(s,A,B,full): super().__init__(); s.A=A; s.B=B; s.full=full
        def forward(s,x,*a,**k):
            o=x+((x@s.A)@s.B if s.B is not None else x@s.A)
            return (o,)
    ndrop=cfg.get("drop",0); patch_bytes=0; dropinfo=None
    if ndrop:
        NL=len(layers); nseq=min(48,len(stream))
        with torch.no_grad():
            if "drop_start" in cfg: a=cfg["drop_start"]
            else:
                dist=torch.zeros(NL+1)
                for i in range(nseq):
                    hs=model.model(stream[i:i+1].cuda(),output_hidden_states=True).hidden_states
                    for l in range(1,NL-ndrop):
                        x=hs[l][0,1:].float(); z=hs[l+ndrop][0,1:].float()
                        c=F.cosine_similarity(x,z,dim=-1).clamp(-1,1)
                        dist[l]+=(torch.arccos(c)/math.pi).mean().item()
                lo=cfg.get("drop_lo",2)
                a=min(range(lo,NL-ndrop),key=lambda l:dist[l].item())
            b=a+ndrop; assert b<=NL-1
            pk=cfg.get("patch","none")
            if pk!="none":
                XtX=torch.zeros(D,D,device="cuda"); XtY=torch.zeros(D,D,device="cuda")
                for i in range(nseq):
                    hs=model.model(stream[i:i+1].cuda(),output_hidden_states=True).hidden_states
                    x=hs[a][0,1:].float(); yy=(hs[b][0,1:].float()-x)
                    XtX+=x.T@x; XtY+=x.T@yy
                lam=cfg.get("lam",1e-3)*XtX.diagonal().mean()
                P=torch.linalg.solve(XtX+lam*torch.eye(D,device="cuda"),XtY)
                if pk=="full":
                    A=P.to(torch.bfloat16); Bm=None; patch_bytes=D*D*2
                else:
                    r=int(pk); G=P.T@XtX@P
                    ev,evec=torch.linalg.eigh(G); V=evec[:,-r:]
                    A=(P@V).to(torch.bfloat16); Bm=V.T.contiguous().to(torch.bfloat16); patch_bytes=2*D*r*2
                newl=list(layers[:a])+[Patch(A,Bm,pk=="full")]+list(layers[b:])
            else:
                newl=list(layers[:a])+list(layers[b:])
            dropinfo=[a,b]
            model.model.layers=nn.ModuleList(newl); layers=model.model.layers
            real=[l for l in layers if hasattr(l,"self_attn")]
            for i,l in enumerate(real): l.self_attn.layer_idx=i
            model.config.num_hidden_layers=len(real)
            print("dropped layers",a,b,"patch",pk,flush=True)
    # ---- sequential per-layer pruning / quant
    ffn_keep=cfg.get("ffn_keep",1.0); head_keep=cfg.get("head_keep",1.0)
    ffn_rec=cfg.get("ffn_rec",True); head_rec=cfg.get("head_rec",True)
    bits=cfg.get("bits",16); g=cfg.get("g",128); lr_rank=cfg.get("lr_rank",0)
    stat=dict(wbits=0.0,wbits_ent=0.0,nparam=0,scale_bits=0.0,lr_bits=0.0,idx_bits=0.0)
    do_proc = ffn_keep<1.0 or head_keep<1.0 or bits<16
    def qparams(x,b):
        n=2**b-1; xmin=x.min(1,keepdim=True)[0]; xmax=x.max(1,keepdim=True)[0]; best=None
        for p in ([1.0,0.95,0.9,0.85,0.8,0.7,0.6,0.5] if b<=3 else [1.0,0.95,0.9,0.85]):
            lo=xmin*p; hi=xmax*p; s=((hi-lo)/n).clamp(min=1e-8); z=torch.round(-lo/s)
            q=(torch.clamp(torch.round(x/s)+z,0,n)-z)*s; e=((q-x)**2).sum(1,keepdim=True)
            if best is None: best=(e,s,z)
            else:
                m=e<best[0]; best=(torch.where(m,e,best[0]),torch.where(m,s,best[1]),torch.where(m,z,best[2]))
        return best[1],best[2]
    def gptq(W,H,b,blocksize=128,damp=0.01):
        W=W.float().clone(); R,C=W.shape; H=H.clone()
        dead=torch.diag(H)==0; H[dead,dead]=1; W[:,dead]=0
        Hd=H+damp*torch.mean(torch.diag(H))*torch.eye(C,device=H.device)
        Hinv=torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(Hd)),upper=True)
        Q=torch.zeros_like(W); hist=torch.zeros(2**b,device=W.device); n=2**b-1; s=z=None
        for i1 in range(0,C,blocksize):
            i2=min(i1+blocksize,C); W1=W[:,i1:i2].clone(); Q1=torch.zeros_like(W1); E1=torch.zeros_like(W1); Hi=Hinv[i1:i2,i1:i2]
            for i in range(i2-i1):
                w=W1[:,i]; d=Hi[i,i]
                if (i1+i)%g==0: s,z=qparams(W1[:,i:i+g],b)
                qi=torch.clamp(torch.round(w.unsqueeze(1)/s)+z,0,n); q=((qi-z)*s).flatten()
                hist+=torch.bincount(qi.long().flatten(),minlength=2**b).float()
                Q1[:,i]=q; err=(w-q)/d
                W1[:,i:]-=err.unsqueeze(1)@Hi[i,i:].unsqueeze(0); E1[:,i]=err
            Q[:,i1:i2]=Q1; W[:,i2:]-=E1@Hinv[i1:i2,i2:]
        return Q,hist,Hd
    def entropy(h):
        p=h/h.sum(); p=p[p>0]; return float(-(p*p.log2()).sum())
    def quant_stack(mods):
        Ws=torch.cat([m.weight.data.float() for m in mods],0)
        return Ws
    def lin(w,bias=None):
        m=nn.Linear(w.shape[1],w.shape[0],bias=bias is not None,device="cuda",dtype=torch.bfloat16)
        m.weight.data=w.to(torch.bfloat16)
        if bias is not None: m.bias.data=bias.to(torch.bfloat16)
        return m
    def ls_recon(H,Wd,keep,lam=0.01):
        # minimize ||X Wd^T - X_S W'^T||  -> W'^T = (H_SS+lam)^-1 H_S,: Wd^T
        Hs=H[keep][:,keep]; Hs=Hs+lam*Hs.diagonal().mean()*torch.eye(len(keep),device=H.device)
        rhs=H[keep]@Wd.float().T
        return torch.linalg.solve(Hs,rhs).T.contiguous()
    cur=None
    if do_proc:
        ins=[]; kw={}
        class Catcher(nn.Module):
            def __init__(s,m): super().__init__(); s.m=m
            def forward(s,x,*a,**k): ins.append(x); kw.update(k); raise ValueError
        first=layers[0]; layers[0]=Catcher(first)
        for i in range(len(stream)):
            try: model(stream[i:i+1].cuda())
            except ValueError: pass
        layers[0]=first
        cur=ins
    def fwd_all(l):
        with torch.no_grad():
            for j,x in enumerate(cur):
                o=l(x,**kw); cur[j]=o[0] if isinstance(o,tuple) else o
    def collect(l,names,skip0=False):
        mods={n:l.get_submodule(n) for n in names}; Hs={}; hds=[]
        for n,m in mods.items():
            Hs[n]=torch.zeros(m.in_features,m.in_features,device="cuda"); 
        cnt={n:0 for n in names}
        def mk(n):
            def hook(m,i,o):
                x=(i[0][:,1:] if skip0 else i[0]).reshape(-1,i[0].shape[-1]).float(); cnt[n]+=x.shape[0]; Hs[n]+=x.T@x
            return hook
        hds=[m.register_forward_hook(mk(n)) for n,m in mods.items()]
        with torch.no_grad():
            for x in cur: l(x,**kw)
        for h_ in hds: h_.remove()
        return {n:Hs[n]/cnt[n] for n in names}
    nreal=0; fixed_params=0; lin_params_kept=0; orig_lin=0
    for li,l in enumerate(layers):
        if not hasattr(l,"self_attn"):
            if do_proc: fwd_all(l)
            continue
        nreal+=1
        at=l.self_attn; ml=l.mlp
        orig_lin+=sum(m.weight.numel() for m in (at.q_proj,at.k_proj,at.v_proj,at.o_proj,ml.gate_proj,ml.up_proj,ml.down_proj))
        if do_proc:
            # --- stage 1: prune
            if ffn_keep<1.0 or head_keep<1.0:
                Hs=collect(l,["self_attn.o_proj","mlp.down_proj"],cfg.get("skip0",False))
                if head_keep<1.0:
                    H=Hs["self_attn.o_proj"]; Wo=at.o_proj.weight.data.float(); ng=at.num_key_value_heads; hd=at.head_dim
                    gw=at.num_heads//ng*hd
                    sc=torch.stack([(H.diagonal()[k*gw:(k+1)*gw]*(Wo[:,k*gw:(k+1)*gw]**2).sum(0)).sum() for k in range(ng)])
                    nk=max(1,int(round(ng*head_keep))); kg=sorted(sc.topk(nk)[1].tolist())
                    cols=torch.cat([torch.arange(k*gw,(k+1)*gw) for k in kg]).cuda()
                    Wn=ls_recon(H,Wo,cols,cfg.get('lam',0.01)) if head_rec else Wo[:,cols]
                    kvr=torch.cat([torch.arange(k*hd,(k+1)*hd) for k in kg]).cuda()
                    at.q_proj=lin(at.q_proj.weight.data[cols],at.q_proj.bias.data[cols])
                    at.k_proj=lin(at.k_proj.weight.data[kvr],at.k_proj.bias.data[kvr])
                    at.v_proj=lin(at.v_proj.weight.data[kvr],at.v_proj.bias.data[kvr])
                    at.o_proj=lin(Wn)
                    at.num_heads=nk*(at.num_heads//ng); at.num_key_value_heads=nk; at.hidden_size=at.num_heads*hd
                    stat["idx_bits"]+=ng
                if ffn_keep<1.0:
                    H=Hs["mlp.down_proj"]; Wd=ml.down_proj.weight.data.float(); I=Wd.shape[1]
                    sc=H.diagonal()*(Wd**2).sum(0); nk=int(round(I*ffn_keep))
                    keep=sorted(sc.topk(nk)[1].tolist()); keep_t=torch.tensor(keep,device="cuda")
                    Wn=ls_recon(H,Wd,keep_t,cfg.get('lam',0.01)) if ffn_rec else Wd[:,keep_t]
                    ml.gate_proj=lin(ml.gate_proj.weight.data[keep_t]); ml.up_proj=lin(ml.up_proj.weight.data[keep_t]); ml.down_proj=lin(Wn)
                    p_=nk/I; stat["idx_bits"]+=I*(-(p_*math.log2(p_)+(1-p_)*math.log2(max(1-p_,1e-9)))) if p_<1 else 0
                del Hs
            # --- stage 2: quantize
            if bits<16:
                Hs=collect(l,["self_attn.q_proj","self_attn.o_proj","mlp.gate_proj","mlp.down_proj"])
                groups=[(["self_attn.q_proj","self_attn.k_proj","self_attn.v_proj"],"self_attn.q_proj"),(["self_attn.o_proj"],"self_attn.o_proj"),
                        (["mlp.gate_proj","mlp.up_proj"],"mlp.gate_proj"),(["mlp.down_proj"],"mlp.down_proj")]
                for names,hn in groups:
                    mods=[l.get_submodule(n) for n in names]; Ws=quant_stack(mods); H=Hs[hn]
                    Q,hist,Hd=gptq(Ws,H,bits)
                    R_,C_=Q.shape; stat["nparam"]+=Q.numel(); stat["wbits"]+=Q.numel()*bits; stat["wbits_ent"]+=Q.numel()*entropy(hist)
                    stat["scale_bits"]+=Q.numel()/g*(16+bits)
                    if lr_rank>0:
                        E=Ws-Q; Lc=torch.linalg.cholesky(Hd)
                        U,S,V=torch.svd_lowrank(E@Lc,q=lr_rank+8,niter=3)
                        U=U[:,:lr_rank]; S=S[:,:lr_rank] if S.dim()>1 else S[:lr_rank]; V=V[:,:lr_rank]
                        A_=(U*S).to(torch.bfloat16).float(); B_=torch.linalg.solve_triangular(Lc.T,V.T,upper=True,left=False).to(torch.bfloat16).float() if False else (V.T@torch.linalg.inv(Lc)).to(torch.bfloat16).float()
                        Q=Q+A_@B_; stat["lr_bits"]+=lr_rank*(R_+C_)*16
                    o=0
                    for m in mods:
                        r_=m.weight.shape[0]; m.weight.data=Q[o:o+r_].to(torch.bfloat16); o+=r_
                    del Q,Ws
                del Hs
            fwd_all(l)
        else: pass
        if nreal%8==0: print("layer",li,time.time()-t0,flush=True)
        lin_params_kept+=sum(m.weight.numel() for m in (at.q_proj,at.k_proj,at.v_proj,at.o_proj,ml.gate_proj,ml.up_proj,ml.down_proj))
        fixed_params+=sum(p.numel() for n,p in l.named_parameters() if not (n.endswith("proj.weight")))
    cur=None; torch.cuda.empty_cache()
    lg=predict(DEV); pred=lg.argmax(1); acc=float((pred==y).mean()); agree=float((pred==base_pred).mean())
    # ---- bytes
    base_lin=48*sum(m for m in [5120*5120,5120*1024,5120*1024,5120*5120,3*5120*13824])
    nlay_eff=nreal-tied_layers   # tied layers share weights
    if bits<16: wb=stat["wbits_ent"]+stat["scale_bits"]; wb_nom=stat["wbits"]+stat["scale_bits"]
    else: wb=wb_nom=lin_params_kept*16
    # tied layers: their params counted zero
    if tied_layers:
        per=lin_params_kept/nreal; wb*=nlay_eff/nreal; wb_nom*=nlay_eff/nreal
    emb_bits=33000*D*4+33000*D/64*(16+4)
    fixed_bits=(fixed_params+D)*16*(nlay_eff/nreal if tied_layers else 1)+4*D*16+0.5e6*8
    tot_ent=wb+stat["lr_bits"]+stat["idx_bits"]+patch_bytes*8+emb_bits+fixed_bits
    tot_nom=wb_nom+stat["lr_bits"]+stat["idx_bits"]+patch_bytes*8+emb_bits+fixed_bits
    res=dict(name=name,cfg=cfg,n=len(DEV),base_acc=base_acc,acc=acc,agree=agree,
        b_wrong=int(((base_pred==y)&(pred!=y)).sum()),c_fixed=int(((base_pred!=y)&(pred==y)).sum()),
        bytes_ent=tot_ent/8,bytes_nom=tot_nom/8,ratio_ent=tot_ent/8/BASE_BYTES,ratio_nom=tot_nom/8/BASE_BYTES,
        body_frac_kept=lin_params_kept/48/(base_lin/48) if not ndrop else lin_params_kept/base_lin,
        bpw_body=(wb/ max(1,lin_params_kept)) , dropped=dropinfo,patch_bytes=patch_bytes,secs=time.time()-t0)
    print("RESULT_JSON "+json.dumps(res),flush=True)
    json.dump(res,open(f"/out/{name}.json","w")); ovol.commit()
    return res

def delta_analysis(model):
    import torch
    L=model.model.layers; out={}
    names=["self_attn.q_proj","self_attn.o_proj","mlp.gate_proj","mlp.up_proj","mlp.down_proj"]
    def rtn_err(W,b=3,g=128):
        W=W.float(); R,C=W.shape; W=W.reshape(R,C//g,g); n=2**b-1
        lo=W.min(-1,keepdim=True)[0]; hi=W.max(-1,keepdim=True)[0]; s=((hi-lo)/n).clamp(min=1e-8); q=torch.round((W-lo)/s)*s+lo
        return float(((q-W)**2).sum()/ (W**2).sum())
    for n in names:
        rel=[];cos=[];e_w=[];e_d=[]
        for i in range(len(L)-1):
            a=L[i].get_submodule(n).weight.data.float(); b=L[i+1].get_submodule(n).weight.data.float()
            d=a-b; rel.append(float(d.norm()/a.norm())); cos.append(float((a*b).sum()/(a.norm()*b.norm())))
            if i%6==0: e_w.append(rtn_err(a)); e_d.append(rtn_err(d))
        out[n]=dict(rel_delta_mean=sum(rel)/len(rel),rel_delta_min=min(rel),cos_mean=sum(cos)/len(cos),cos_max=max(cos),rtn3_relerr_W=sum(e_w)/len(e_w),rtn3_relerr_delta=sum(e_d)/len(e_d))
    return {"delta":out}

@app.function(image=img, timeout=86400, volumes={"/out":ovol})
def driver(cfgs):
    rs=[]
    for r in run.map(cfgs,return_exceptions=True):
        rs.append(r if not isinstance(r,Exception) else {"error":repr(r)})
        print("DONE",json.dumps(rs[-1])[:300],flush=True)
    json.dump(rs,open(f"/out/batch_{cfgs[0]['name']}.json","w")); ovol.commit()
    return rs

@app.local_entrypoint()
def main(cfgs:str):
    cs=json.load(open(cfgs))
    fc=driver.spawn(cs)
    print("spawned driver",fc.object_id,len(cs))
