"""Task C (tool use) compression sweep. Qwen2.5 base -> [layer drop] -> Hadamard+GPTQ L-level entropy-coded -> vocab/head trim + low-bit emb.
Eval: generative exact tool-call match on dev (first nev); outputs saved for agreement with base. Calibration: C/calib only. Never touches locked.
Launch: modal run --detach cs.py --cfgs cfgs.json  -> RESULT_JSON in app logs + /res/<name>.json in volume tenpct-C-res"""
import modal, json
app = modal.App("tenpct-C")
img = (modal.Image.debian_slim().pip_install("torch","transformers==4.51.3","numpy","accelerate")
       .add_local_file("../commercial/C/dev.jsonl","/root/C_dev.jsonl").add_local_file("../commercial/C/calib.jsonl","/root/C_calib.jsonl")
       .add_local_file("../commercial/C/pool.jsonl","/root/C_pool.jsonl"))
vol = modal.Volume.from_name("hf-cache", create_if_missing=True)
rvol = modal.Volume.from_name("tenpct-C-res", create_if_missing=True)

@app.function(image=img, gpu=["A100-80GB","H100"], timeout=14400, max_containers=60, volumes={"/hf":vol,"/res":rvol})
def run(cfg):
    import os, sys, time, math, torch, numpy as np, torch.nn as nn
    os.environ["HF_HOME"]="/hf"
    from transformers import AutoTokenizer, AutoModelForCausalLM, LogitsProcessor, LogitsProcessorList
    t0=time.time()
    M=cfg.get("model","Qwen/Qwen2.5-7B-Instruct"); mode="eval"
    tok=AutoTokenizer.from_pretrained(M); tok.padding_side="left"
    model=AutoModelForCausalLM.from_pretrained(M,torch_dtype=torch.bfloat16).cuda().eval()
    model.config.use_cache=False
    base_params=sum(p.numel() for p in model.parameters())
    rd=lambda f:[json.loads(x) for x in open("/root/C_"+f)]
    calsrc=rd("calib.jsonl"); DEV=rd("dev.jsonl"); POOL=rd("pool.jsonl")
    SYS="You are a function-calling assistant. Available tools (JSON):\n%s\nRespond ONLY with a JSON list of tool calls in the form [{\"name\": ..., \"arguments\": {...}}]."
    def prompt(r):
        return tok.apply_chat_template([{"role":"system","content":SYS%r["tools"]},{"role":"user","content":r["text"]}],add_generation_prompt=True,tokenize=False)
    def full(r): return prompt(r)+r["gold_calls"]+"<|im_end|>"
    ev=DEV[:cfg.get("nev",300)]
    other=POOL
    rng=np.random.RandomState(0)
    cal_txt=[full(r) for r in calsrc]; rng.shuffle(cal_txt)
    ncal=cfg.get("ncal",128); L_=cfg.get("callen",1024)
    stream=[]
    for s_ in cal_txt: stream+=tok.encode(s_)
    stream=torch.tensor(stream[:ncal*L_]).view(-1,L_)[:ncal]
    V=model.get_input_embeddings().weight.shape[0]; keepN=cfg.get("vocab",0)
    remap=lambda ids: ids; oov=[0,0]; nkeep=V; km=torch.ones(V,dtype=torch.bool)
    if keepN:
        cnt=torch.zeros(V)
        for r in other[:3000]+calsrc:
            ids=tok.encode(full(r)); cnt.index_add_(0,torch.tensor(ids),torch.ones(len(ids)))
        keep=set(cnt.topk(keepN)[1].tolist())
        vinv=tok.convert_ids_to_tokens(list(range(len(tok))))
        for t_,i in tok.get_vocab().items():
            if len(t_)==1: keep.add(i)
        keep|=set(tok.all_special_ids)|set(tok.encode("<|im_start|><|im_end|>"))
        km=torch.zeros(V,dtype=torch.bool); km[list(keep)]=True; nkeep=int(km.sum())
        kstr={vinv[i]:i for i in keep if i<len(vinv)}; maxl=max(len(k) for k in kstr)
        def remap(ids):
            out=[]
            for i in ids:
                oov[1]+=1
                if i<V and km[i]: out.append(i); continue
                oov[0]+=1
                t_=vinv[i] if i<len(vinv) else None
                if t_ is None: out.append(i); continue
                p=0
                while p<len(t_):
                    for q in range(min(len(t_),p+maxl),p,-1):
                        if t_[p:q] in kstr: out.append(kstr[t_[p:q]]); p=q; break
                    else: out.append(i); p+=1
            return out
    class Mask(LogitsProcessor):
        def __call__(self,ids,scores):
            m=km.to(scores.device); scores[:,:len(m)]=scores[:,:len(m)].masked_fill(~m,-1e9); return scores
    @torch.no_grad()
    def gen(rows,rm,bs=cfg.get("bs",12)):
        enc=[rm(tok.encode(prompt(r)))[-6000:] for r in rows]
        order=np.argsort([-len(e) for e in enc]); outs=[None]*len(rows)
        model.config.use_cache=True
        for b in range(0,len(rows),bs):
            idx=order[b:b+bs]; mx=max(len(enc[i]) for i in idx)
            ids=torch.full((len(idx),mx),tok.pad_token_id,dtype=torch.long); att=torch.zeros(len(idx),mx,dtype=torch.long)
            for j,i in enumerate(idx):
                ids[j,mx-len(enc[i]):]=torch.tensor(enc[i]); att[j,mx-len(enc[i]):]=1
            o=model.generate(input_ids=ids.cuda(),attention_mask=att.cuda(),max_new_tokens=cfg.get("maxnew",256),do_sample=False,logits_processor=LogitsProcessorList([Mask()]) if keepN else None,pad_token_id=tok.pad_token_id)
            for j,i in enumerate(idx): outs[i]=tok.decode(o[j,mx:],skip_special_tokens=True).strip()
        model.config.use_cache=False
        return outs
    def norm(cs): return sorted(json.dumps({"name":c["name"],"arguments":{k:str(v) for k,v in c.get("arguments",{}).items()}},sort_keys=True) for c in cs)
    def parse(txt):
        if txt.startswith("```"): txt=txt.strip("`").lstrip("json").strip()
        try: return json.loads(txt)
        except Exception: return None
    def score(outs):
        ok=[];nm=[]
        for r,t in zip(ev,outs):
            gold=json.loads(r["gold_calls"]); pc=parse(t)
            try: ok.append(int(norm(pc)==norm(gold))); nm.append(int(sorted(c["name"] for c in pc)==sorted(c["name"] for c in gold)))
            except Exception: ok.append(0); nm.append(0)
        return ok,nm
    layers=model.model.layers; nl0=len(layers)
    def get_inps(n):
        ins=[];kw={}
        class C(nn.Module):
            def __init__(s,m): super().__init__(); s.m=m
            def forward(s,x,*a,**k): ins.append(x); kw.update(k); raise ValueError
        layers[0]=C(layers[0])
        for i in range(n):
            try: model(stream[i:i+1].cuda())
            except ValueError: pass
        layers[0]=layers[0].m
        return ins,kw
    inps,kw=get_inps(len(stream))
    # ---- layer drop (block influence)
    ndrop=cfg.get("drop",0); dropl=[]
    if ndrop:
        bi=torch.zeros(nl0); cur=[x.clone() for x in inps]
        with torch.no_grad():
            for li,l in enumerate(layers):
                s=0
                for j,x in enumerate(cur):
                    o=l(x,**kw); o=o[0] if isinstance(o,tuple) else o
                    s+=1-torch.nn.functional.cosine_similarity(x.float().flatten(0,1),o.float().flatten(0,1),dim=-1).mean().item(); cur[j]=o
                bi[li]=s/len(cur)
        dropl=sorted(sorted(range(2,nl0-1),key=lambda i:bi[i].item())[:ndrop])
    orig_idx=[i for i in range(nl0) if i not in dropl]
    if dropl:
        model.model.layers=nn.ModuleList([layers[i] for i in orig_idx]); layers=model.model.layers
        for ni,l in enumerate(layers): l.self_attn.layer_idx=ni
        model.config.num_hidden_layers=len(layers)
    # ---- quantization primitives: L-level symmetric grid, per-group fp16 scale
    g=cfg.get("g",128); rot=cfg.get("rot",True)
    PGRID=torch.tensor([1.0,0.9,0.8,0.7,0.62,0.55,0.5,0.45,0.4,0.36,0.32,0.28,0.25]).cuda()
    def qparams(x,Lv):
        half=(Lv-1)/2; am=x.abs().max(1,keepdim=True)[0].clamp(min=1e-8)
        best=None
        for p in PGRID:
            s=(am*p/half).half().float().clamp(min=1e-8)
            q=(torch.clamp(torch.round(x/s+half),0,Lv-1)-half)*s
            e=((q-x)**2).sum(1,keepdim=True)
            if best is None: best=(e,s)
            else:
                m=e<best[0]; best=(torch.where(m,e,best[0]),torch.where(m,s,best[1]))
        return best[1]
    def qidx(x,s,Lv):
        half=(Lv-1)/2; qi=torch.clamp(torch.round(x/s+half),0,Lv-1); return (qi-half)*s,qi
    HB=128
    def hada(n):
        Hm=torch.ones(1,1)
        while Hm.shape[0]<n: Hm=torch.cat([torch.cat([Hm,Hm],1),torch.cat([Hm,-Hm],1)],0)
        return (Hm/math.sqrt(n)).cuda()
    Hb=hada(HB)
    def rot_cols(M,sg):  # M @ Q, Q=diag(sg) blockdiag(Hb)
        R,C=M.shape; return ((M*sg).view(R,C//HB,HB)@Hb).reshape(R,C)
    def unrot_cols(M,sg):  # M @ Q^T
        R,C=M.shape; return ((M.view(R,C//HB,HB)@Hb).reshape(R,C))*sg
    def gptq(W,H,Lv,blocksize=128,damp=0.01):
        W=W.float().clone(); R,C=W.shape; H=H.clone()
        dead=torch.diag(H)==0; H[dead,dead]=1; W[:,dead]=0
        H=H+damp*torch.mean(torch.diag(H))*torch.eye(C,device=H.device)
        Hinv=torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(H)),upper=True)
        Q=torch.zeros_like(W); hist=torch.zeros(Lv,device=W.device); s=None
        for i1 in range(0,C,blocksize):
            i2=min(i1+blocksize,C); W1=W[:,i1:i2].clone(); Q1=torch.zeros_like(W1); Err1=torch.zeros_like(W1); Hi=Hinv[i1:i2,i1:i2]
            for i in range(i2-i1):
                if (i1+i)%g==0:
                    s=qparams(torch.cat([W1[:,i:],W[:,i2:]],1)[:,:g],Lv)
                w=W1[:,i]; d=Hi[i,i]
                q,qi=qidx(w.unsqueeze(1),s,Lv); q=q.flatten()
                hist+=torch.bincount(qi.long().flatten(),minlength=Lv).float()
                Q1[:,i]=q; err=(w-q)/d
                W1[:,i:]-=err.unsqueeze(1)@Hi[i,i:].unsqueeze(0); Err1[:,i]=err
            Q[:,i1:i2]=Q1; W[:,i2:]-=Err1@Hinv[i1:i2,i2:]
        return Q,hist
    def ent(hist):
        p=hist/hist.sum(); p=p[p>0]; return float(-(p*p.log2()).sum())
    linn=["self_attn.q_proj","self_attn.k_proj","self_attn.v_proj","self_attn.o_proj","mlp.gate_proj","mlp.up_proj","mlp.down_proj"]
    groups=[("self_attn.q_proj",["self_attn.q_proj","self_attn.k_proj","self_attn.v_proj"]),("self_attn.o_proj",["self_attn.o_proj"]),
            ("mlp.gate_proj",["mlp.gate_proj","mlp.up_proj"]),("mlp.down_proj",["mlp.down_proj"])]
    levels=cfg.get("levels",{}); defL=cfg.get("L",0)   # defL=0 -> unquantized
    LO,HI=cfg.get("lay_lo",0),cfg.get("lay_hi",10**9)
    stats=dict(nq=0,ent_bits=0.0,fix_bits=0.0,scale_bits=0.0); table={}; layer_bits=[]
    def Lfor(oli,n): return int(levels.get(f"{oli}.{n}",defL))
    quantize_on = (mode=="table") or defL>0 or len(levels)>0
    if quantize_on:
        cur=[x.clone() for x in inps]
        for ni,l in enumerate(layers):
            oli=orig_idx[ni]
            if mode=="table" and not (LO<=oli<HI):
                with torch.no_grad():
                    for j,x in enumerate(cur):
                        o=l(x,**kw); cur[j]=o[0] if isinstance(o,tuple) else o
                continue
            mods={n:l.get_submodule(n) for n in linn}
            Hs={}; nsm={}
            def mkh(key):
                def hook(m,i,o):
                    x=i[0].reshape(-1,i[0].shape[-1]).float(); nsm[key]=nsm.get(key,0)+x.shape[0]
                    Hs[key]=Hs.get(key,0)+2*x.T@x
                return hook
            hds=[mods[k].register_forward_hook(mkh(k)) for k,_ in groups]
            with torch.no_grad():
                for x in cur: l(x,**kw)
            for hh in hds: hh.remove()
            lb=0.0
            for key,members in groups:
                H=Hs[key]/nsm[key]; C=H.shape[0]
                sg=(torch.randint(0,2,(C,),generator=torch.Generator().manual_seed(1000*oli+len(key)),dtype=torch.float32)*2-1).cuda() if rot else torch.ones(C).cuda()
                if rot:
                    H1=rot_cols(H,sg); Hr=rot_cols(H1.T.contiguous(),sg).T.contiguous()
                else: Hr=H
                for n in members:
                    m=mods[n]; W=m.weight.data.float(); Wr=rot_cols(W,sg) if rot else W
                    if mode=="table":
                        ref=float(((Wr@Hr)*Wr).sum())
                        tab={}
                        for Lv in cfg.get("table_levels",[2,3,4,6,8,16]):
                            Q,hist=gptq(Wr,Hr,Lv); D=Wr-Q
                            tab[Lv]=[float(((D@Hr)*D).sum())/ref, ent(hist)]
                        table[f"{oli}.{n}"]=dict(n=W.numel(),tab=tab)
                    else:
                        Lv=Lfor(oli,n)
                        if Lv<=0: stats["fix_bits"]+=W.numel()*16; lb+=W.numel()*16; continue
                        Q,hist=gptq(Wr,Hr,Lv)
                        What=unrot_cols(Q,sg) if rot else Q
                        m.weight.data=What.to(torch.bfloat16)
                        nn_=W.numel(); stats["nq"]+=nn_; e_=ent(hist)
                        stats["ent_bits"]+=nn_*e_; stats["scale_bits"]+=nn_/g*16; lb+=nn_*(e_+16/g)
            del Hs
            layer_bits.append(lb/ sum(mods[n].weight.numel() for n in linn))
            with torch.no_grad():
                for j,x in enumerate(cur):
                    o=l(x,**kw)
                    cur[j]=o[0] if isinstance(o,tuple) else o   # table mode: layer unmodified -> fp propagation
            print("layer",oli,round(time.time()-t0),flush=True)
    if mode=="table":
        out=dict(model=M,rot=rot,g=g,lay_lo=LO,lay_hi=HI,table=table)
        os.makedirs("/hf/v2tables",exist_ok=True); json.dump(out,open(f"/hf/v2tables/{M.replace('/','_')}_rot{int(rot)}_g{g}_{LO}_{HI}.json","w")); vol.commit()
        print("TABLE_JSON "+json.dumps(out),flush=True); return out
    # embedding quant (+ untied head)
    emb=model.get_input_embeddings(); hdim=emb.weight.shape[1]; ebits=cfg.get("ebits",16)
    tied=model.config.tie_word_embeddings
    def qrows(wt):
        w=wt.data.float(); wq=torch.zeros_like(w); Lv=2**ebits
        for c in range(0,w.shape[1],64):
            blk=w[:,c:c+64]; s_=qparams(blk,Lv); wq[:,c:c+64]=qidx(blk,s_,Lv)[0]
        wt.data=wq.to(torch.bfloat16)
    if ebits<16:
        qrows(emb.weight)
        if not tied: qrows(model.get_output_embeddings().weight)
    outs=gen(ev,remap); ok,nm=score(outs)
    lin_total=sum(l.get_submodule(n).weight.numel() for l in layers for n in linn)
    other_layer=sum(p.numel() for l in layers for p in l.parameters())-lin_total
    if not quantize_on: stats["fix_bits"]=lin_total*16
    anso=cfg.get("ans_overhead",0.005)
    body_bits=stats["ent_bits"]*(1+anso)+stats["scale_bits"]+stats["fix_bits"]
    ebw=16 if ebits>=16 else ebits+16/64
    emb_bits=nkeep*hdim*ebw*(1 if tied else 2)
    misc_bits=(other_layer+model.model.norm.weight.numel())*16+cfg.get("slack_bytes",1_000_000)*8
    tot=body_bits+emb_bits+misc_bits; base_bf16=base_params*2
    res=dict(name=cfg.get("name"),cfg={k:v for k,v in cfg.items() if k!="levels"},n=len(ev),exact=float(np.mean(ok)),name_match=float(np.mean(nm)),ok=ok,
        params=base_params,nkeep=nkeep,dropped=dropl,bytes=tot/8,ratio_bf16=tot/8/base_bf16,
        body_bpw=(body_bits/max(1,lin_total)),parts_MB=dict(body=body_bits/8e6,emb=emb_bits/8e6,misc=misc_bits/8e6),
        oov=oov[0]/max(1,oov[1]),secs=time.time()-t0)
    os.makedirs("/res",exist_ok=True)
    json.dump(dict(res=res,outs=outs),open(f"/res/{cfg.get('name','x')}.json","w")); rvol.commit()
    print("RESULT_JSON "+json.dumps(res,default=str),flush=True); return res

@app.local_entrypoint()
def main(cfgs:str):
    import time
    cs=json.load(open(cfgs)); hs=[]
    for c in cs:
        if "levels_file" in c: c["levels"]=json.load(open(c["levels_file"]))
        hs.append(run.spawn(c)); time.sleep(c.get("stagger",15))
    for hnd in hs:
        try: hnd.get()
        except Exception as e: print("ERR",repr(e),flush=True)
