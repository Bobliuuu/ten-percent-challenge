"""C-DENSE: no-training PTQ of dense Qwen bases toward 10% of bf16 (Modal). Rotation(Hadamard)+GPTQ onto L-level entropy-coded
grids, per-module L from RTN sensitivity tables (Lagrangian), optional BI layer drop, embedding+lm_head quantised (never trimmed).
Fidelity vs bf16 base: greedy 64-token agreement, teacher-forced top1 + KL (top-64 + tail bucket). NO optimizer/backward anywhere.
Launch: modal run --detach dense.py --cfgs cfgs.json --out out.json"""
import modal, json
app = modal.App("tenpct-dense2")
img = (modal.Image.debian_slim().pip_install("torch==2.4.1","transformers==4.55.0","numpy","accelerate","huggingface_hub")
       .add_local_file("data/dense_data.json","/root/dense_data.json"))
vol = modal.Volume.from_name("tenpct-v2-hf", create_if_missing=True)
SIZES = {"s":dict(gpu="A100-80GB",memory=64000),"m":dict(gpu=["H100","A100-80GB"],memory=120000),"l":dict(gpu="A100-80GB",memory=180000)}

def _run(cfg):
    if 'multi' in cfg:
        out=[]
        for c in cfg['multi']:
            try: out.append(_run(c))
            except Exception as e:
                import traceback; traceback.print_exc(); out.append({'error':repr(e),'cfg':c})
        return out
    import os, sys, time, math, torch, numpy as np, torch.nn as nn
    HFD=os.environ.get("DENSE_HF","/hf"); os.environ.setdefault("HF_HOME","/hf") if HFD=="/hf" else None
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from huggingface_hub import HfApi
    t0=time.time(); torch.manual_seed(0)
    M=cfg["model"]; tag=M.split("/")[-1]; mode=cfg.get("mode","eval"); name=cfg.get("name","x")
    D=json.load(open(os.environ.get("DENSE_DATA","/root/dense_data.json"))); os.makedirs(HFD+"/dense",exist_ok=True)
    tok=AutoTokenizer.from_pretrained(M); tok.padding_side="left"
    if tok.pad_token is None: tok.pad_token=tok.eos_token
    def chat(p):
        try: return tok.apply_chat_template([{"role":"user","content":p}],add_generation_prompt=True,tokenize=False,enable_thinking=False)
        except TypeError: return tok.apply_chat_template([{"role":"user","content":p}],add_generation_prompt=True,tokenize=False)
    def enc_prompt(p): return tok.encode(chat(p),add_special_tokens=False)[-900:]
    NEW=cfg.get('max_new',64)
    GPUR=bool(os.environ.get('DENSE_GPU'))
    model=AutoModelForCausalLM.from_pretrained(M,torch_dtype=torch.bfloat16,low_cpu_mem_usage=True,**(dict(device_map='cuda') if GPUR else {})).eval()
    model.config.use_cache=False
    base_params=sum(p.numel() for p in model.parameters()); base_bytes=base_params*2
    ev_id,ev_ood,tfs=D["eval_id"],D["eval_ood"],D["tf"]
    eos=set([tok.eos_token_id]+([model.generation_config.eos_token_id] if isinstance(model.generation_config.eos_token_id,int) else list(model.generation_config.eos_token_id or [])))
    @torch.no_grad()
    def gen_all(prompts,bs=32):
        enc=[enc_prompt(p) for p in prompts]; order=np.argsort([-len(e) for e in enc]); out=[None]*len(enc)
        for b in range(0,len(enc),bs):
            idx=order[b:b+bs]; mx=max(len(enc[i]) for i in idx)
            ids=torch.full((len(idx),mx),tok.pad_token_id); att=torch.zeros(len(idx),mx,dtype=torch.long)
            for j,i in enumerate(idx): ids[j,mx-len(enc[i]):]=torch.tensor(enc[i]); att[j,mx-len(enc[i]):]=1
            g=model.generate(input_ids=ids.cuda(),attention_mask=att.cuda(),max_new_tokens=NEW,do_sample=False,use_cache=True,pad_token_id=tok.pad_token_id,temperature=None,top_p=None,top_k=None)
            g=g[:,mx:].cpu().tolist()
            for j,i in enumerate(idx):
                s=g[j]
                for k,t in enumerate(s):
                    if t in eos: s=s[:k+1]; break
                out[i]=s
        return out
    @torch.no_grad()
    def tf_logp(items,K=64):   # items: (prompt_ids, resp_ids) -> per item (ids[P,K], lp[P,K], lse info)
        res=[]
        for pi,ri in items:
            ids=torch.tensor([(pi+ri)[-640:]]).cuda(); P=len(ri)
            lg=model(input_ids=ids).logits[0,-P-1:-1].float()
            lp=torch.log_softmax(lg,-1); tv,ti=lp.topk(K,-1)
            res.append((ti.cpu().numpy().astype(np.int32),tv.cpu().numpy().astype(np.float32)))
        return res
    def tf_items():
        A=[(enc_prompt(t["prompt"]),tok.encode(t["ref"],add_special_tokens=False)[:150]+[tok.convert_tokens_to_ids("<|im_end|>")]) for t in tfs]
        return A
    bpath=f"{HFD}/dense/base_{tag}.json"
    def kind_masks(ev): return {k:[i for i,e in enumerate(ev) if e["kind"]==k] for k in sorted(set(e["kind"] for e in ev))}

    def t1_items():
        T1=os.environ.get("DENSE_T1","/root/t1"); sys.path.insert(0,T1)
        return [json.loads(l) for l in open(T1+"/probe_items.jsonl")]
    @torch.no_grad()
    def gen_raw(texts,newtok,bs=16):
        enc=[tok.encode(t,add_special_tokens=False)[-12000:] for t in texts]; order=np.argsort([-len(e) for e in enc]); out=[None]*len(enc)
        for b in range(0,len(enc),bs):
            idx=order[b:b+bs]; mx=max(len(enc[i]) for i in idx)
            ids=torch.full((len(idx),mx),tok.pad_token_id); att=torch.zeros(len(idx),mx,dtype=torch.long)
            for j,i in enumerate(idx): ids[j,mx-len(enc[i]):]=torch.tensor(enc[i]); att[j,mx-len(enc[i]):]=1
            g=model.generate(input_ids=ids.cuda(),attention_mask=att.cuda(),max_new_tokens=newtok,do_sample=False,use_cache=True,pad_token_id=tok.pad_token_id,temperature=None,top_p=None,top_k=None)
            for j,i in enumerate(idx): out[i]=tok.decode(g[j,mx:],skip_special_tokens=False)
        return out
    T1RES={}
    def run_t1():
        items=t1_items()
        from scorer import score_item
        kw={"enable_thinking":False}
        prompts=[]
        for it in items:
            tools=[{"type":"function","function":t} for t in it["tools"]]
            try: prompts.append(tok.apply_chat_template([{"role":"user","content":it["query"]}],tools=tools,add_generation_prompt=True,tokenize=False,**kw))
            except TypeError: prompts.append(tok.apply_chat_template([{"role":"user","content":it["query"]}],tools=tools,add_generation_prompt=True,tokenize=False))
        model.config.use_cache=True
        txt=gen_raw(prompts,cfg.get("t1_new",400),bs=cfg.get("t1_bs",16)); rows=[]
        for it,t in zip(items,txt):
            s_=score_item(t,it); rows.append(dict(id=it["id"],source=it["source"],kind=it["kind"],correct=bool(s_["correct"]),parse_fail=bool(s_["parse_fail"]),output=t[:600]))
        T1RES["rows"]=rows
    def t1_result(locals_):
        d=dict(cfg=cfg,model=M,t1=T1RES["rows"],secs=time.time()-t0)
        if locals_:
            tb=locals_["tot_bytes"]; d.update(bytes=tb,base_bytes=locals_["base_bytes"],ratio=tb/locals_["base_bytes"],dropped=locals_["dropped"],body_bpw=locals_["qbits"]/locals_["lin_total"])
        json.dump(d,open(f"{HFD}/dense/t1_{tag}_{name}.json","w"),default=str); (vol.commit() if HFD=='/hf' else None); print("T1_DONE",tag,name,sum(r["correct"] for r in d["t1"]),flush=True); return d
    if mode=="t1base":
        model.cuda(); run_t1(); return t1_result(None)
    if mode=="base":
        model.cuda()
        gid=gen_all([e["prompt"] for e in ev_id]); god=gen_all([e["prompt"] for e in ev_ood])
        g2=gen_all([e["prompt"] for e in ev_id[::4]],bs=8)   # numeric-noise floor: same model, different batch size
        floor=float(np.mean([a[:NEW]==b[:NEW] for a,b in zip(gid[::4],g2)]))
        sel=[i for k,v in kind_masks(ev_ood).items() for i in v[:25]]
        oi=[(enc_prompt(ev_ood[i]["prompt"]),god[i]) for i in sel]
        tfi=tf_logp(tf_items()); tfo=tf_logp(oi)
        out=dict(gid=gid,god=god,floor_bs32_vs_bs8=floor,ood_sel=sel,
                 tf=[(a.tolist(),b.tolist()) for a,b in tfi],tfo=[(a.tolist(),b.tolist()) for a,b in tfo],base_bytes=base_bytes,base_params=base_params)
        json.dump(out,open(bpath,"w")); (vol.commit() if HFD=='/hf' else None); print("BASE_DONE",tag,"floor",floor,"secs",round(time.time()-t0),flush=True)
        return dict(tag=tag,floor=floor,base_bytes=base_bytes)
    # ================= compression =================
    base=json.load(open(bpath))
    g=cfg.get("g",256); GB=cfg.get("scale_bits",11); dz=cfg.get("dz",0.0); HB=128
    layers=model.model.layers; nl0=len(layers)
    linn=["self_attn.q_proj","self_attn.k_proj","self_attn.v_proj","self_attn.o_proj","mlp.gate_proj","mlp.up_proj","mlp.down_proj"]
    groups=[("self_attn.q_proj",["self_attn.q_proj","self_attn.k_proj","self_attn.v_proj"]),("self_attn.o_proj",["self_attn.o_proj"]),("mlp.gate_proj",["mlp.gate_proj","mlp.up_proj"]),("mlp.down_proj",["mlp.down_proj"])]
    cal_txt=[chat(c["prompt"])+c["ref"]+"<|im_end|>\n" for c in D["calib"]]
    if cfg.get("calib_t1"):
        from_t1=[json.loads(l) for l in open(os.environ.get("DENSE_T1","/root/t1")+"/calib_t1.jsonl")]
        cal_txt=[]
        sys.path.insert(0,os.environ.get("DENSE_T1","/root/t1"))
        for it in from_t1:
            tools=[{"type":"function","function":t} for t in it["tools"]]
            try: pr=tok.apply_chat_template([{"role":"user","content":it["query"]}],tools=tools,add_generation_prompt=True,tokenize=False,enable_thinking=False)
            except TypeError: pr=tok.apply_chat_template([{"role":"user","content":it["query"]}],tools=tools,add_generation_prompt=True,tokenize=False)
            ans="\n".join("<tool_call>\n"+json.dumps(dict(name=g["name"],arguments=g["arguments"]))+"\n</tool_call>" for g in it["gold"]) or "None of the available tools applies to this request."
            cal_txt.append(pr+ans+"<|im_end|>\n")
        if cfg.get("calib_mix"): cal_txt=cal_txt[:len(cal_txt)//2]+[chat(c["prompt"])+c["ref"]+"<|im_end|>\n" for c in D["calib"]][:len(cal_txt)//2]
    rng=np.random.RandomState(0); rng.shuffle(cal_txt)
    ncal=cfg.get("ncal",128); CL=cfg.get("CL",512); stream=[]
    for s in cal_txt:
        stream+=tok.encode(s,add_special_tokens=False)
        if len(stream)>=ncal*CL: break
    stream=torch.tensor(stream[:ncal*CL]).view(-1,CL)
    PG=torch.tensor([1.0,.9,.8,.7,.62,.55,.5,.45,.4,.36,.32,.28,.25,.22,.19,.16,.13]).cuda()
    def quant(x,s,Lv,dzv):
        half=(Lv-1)/2; t=x/s
        if Lv%2==1 and dzv>0:
            v=torch.sign(t)*torch.floor(t.abs()+0.5-dzv); v=v.clamp(-half,half); return v*s,(v+half)
        qi=torch.clamp(torch.round(t+half),0,Lv-1); return (qi-half)*s,qi
    def qparams(x,Lv):   # x [N,g] -> scale [N,1]
        half=(Lv-1)/2; am=x.abs().amax(1,keepdim=True).clamp(min=1e-8); best=None
        for p in PG:
            s=(am*p/max(half,0.5)).half().float().clamp(min=1e-8); q=quant(x,s,Lv,dz)[0]; e=((q-x)**2).sum(1,keepdim=True)
            if best is None: best=(e,s)
            else: m=e<best[0]; best=(torch.where(m,e,best[0]),torch.where(m,s,best[1]))
        return best[1]
    Hm=torch.ones(1,1)
    while Hm.shape[0]<HB: Hm=torch.cat([torch.cat([Hm,Hm],1),torch.cat([Hm,-Hm],1)],0)
    Hb=(Hm/math.sqrt(HB)).cuda()
    def rot(Mx,sg): R,C=Mx.shape; return ((Mx*sg).view(R,C//HB,HB)@Hb).reshape(R,C)
    def unrot(Mx,sg): R,C=Mx.shape; return ((Mx.view(R,C//HB,HB)@Hb).reshape(R,C))*sg
    if not cfg.get('rot',True):
        rot=lambda Mx,sg: Mx.clone(); unrot=lambda Mx,sg: Mx.clone()
    def ent(h): p=h/h.sum(); p=p[p>0]; return float(-(p*p.log2()).sum())
    def rtn(W,Lv):   # W [R,C] rotated; returns Q, hist
        R,C=W.shape; out=torch.empty_like(W); hist=torch.zeros(Lv,device=W.device)
        ch=max(1,int(2e7//C))
        for r in range(0,R,ch):
            w=W[r:r+ch]; x=w.reshape(-1,g); s=qparams(x,Lv); q,qi=quant(x,s,Lv,dz); out[r:r+ch]=q.view_as(w); hist+=torch.bincount(qi.long().flatten(),minlength=Lv).float()
        return out,hist
    def gptq(W,H,Lv,damp=0.01):
        if cfg.get('actorder'):
            perm=torch.argsort(torch.diag(H),descending=True); inv=torch.argsort(perm)
            Q,h=gptq0(W[:,perm],H[perm][:,perm],Lv,damp); return Q[:,inv],h
        return gptq0(W,H,Lv,damp)
    def gptq0(W,H,Lv,damp=0.01):
        W=W.float().clone(); R,C=W.shape; H=H.clone(); bsz=max(128,g)
        dead=torch.diag(H)==0; H[dead,dead]=1; W[:,dead]=0
        H=H+damp*torch.mean(torch.diag(H))*torch.eye(C,device=H.device)
        Hinv=torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(H)),upper=True)
        Q=torch.zeros_like(W); hist=torch.zeros(Lv,device=W.device); s=None
        for i1 in range(0,C,bsz):
            i2=min(i1+bsz,C); W1=W[:,i1:i2].clone(); Q1=torch.zeros_like(W1); E1=torch.zeros_like(W1); Hi=Hinv[i1:i2,i1:i2]
            for i in range(i2-i1):
                if i%g==0: s=qparams(W1[:,i:i+g],Lv)
                w=W1[:,i]; d=Hi[i,i]; q,qi=quant(w.unsqueeze(1),s,Lv,dz); q=q.flatten()
                hist+=torch.bincount(qi.long().flatten(),minlength=Lv).float()
                Q1[:,i]=q; err=(w-q)/d; W1[:,i:]-=err.unsqueeze(1)@Hi[i,i:].unsqueeze(0); E1[:,i]=err
            Q[:,i1:i2]=Q1; W[:,i2:]-=E1@Hinv[i1:i2,i2:]
        return Q,hist
    def cuda_kw(kw):
        o={}
        for k,v in kw.items():
            if k in("past_key_values","past_key_value","use_cache"): continue
            if torch.is_tensor(v): o[k]=v.cuda()
            elif isinstance(v,tuple): o[k]=tuple(x.cuda() if torch.is_tensor(x) else x for x in v)
            else: o[k]=v
        return o
    ins=[];kw={}
    class Cap(nn.Module):
        def __init__(s,m): super().__init__(); s.m=m; s.attention_type=getattr(m,'attention_type','full_attention')
        def forward(s,x,*a,**k): ins.append(x); kw.update(k); raise ValueError
    layers[0]=Cap(layers[0])
    with torch.no_grad():
        for i in range(len(stream)):
            try: model.model(input_ids=stream[i:i+1].cuda() if GPUR else stream[i:i+1])
            except ValueError: pass
    layers[0]=layers[0].m; kw=cuda_kw(kw); inps=[x.cuda() for x in ins]; del ins
    def fwd(l,x):
        o=l(x,**kw); return o[0] if isinstance(o,tuple) else o
    def collect(l,cur):
        Hs={};nsm={}
        mods={n:l.get_submodule(n) for n in linn}
        def mk(key):
            def hook(m,i,o):
                x=i[0].reshape(-1,i[0].shape[-1]).float(); nsm[key]=nsm.get(key,0)+x.shape[0]; Hs[key]=Hs.get(key,0)+2*x.T@x
            return hook
        hds=[mods[k].register_forward_hook(mk(k)) for k,_ in groups]
        with torch.no_grad():
            for x in cur: fwd(l,x)
        for h in hds: h.remove()
        return mods,{k:Hs[k]/nsm[k] for k in Hs}
    def collect2(l,curq,curf):
        mods={n:l.get_submodule(n) for n in linn}; store={}; mode=['q']; H={};Cx={};nsm={}
        def mk(key):
            def hook(m,i,o): store[(mode[0],key)]=i[0].reshape(-1,i[0].shape[-1]).float()
            return hook
        hds=[mods[k].register_forward_hook(mk(k)) for k,_ in groups]
        with torch.no_grad():
            for xq,xf in zip(curq,curf):
                mode[0]='q'; fwd(l,xq); mode[0]='f'; fwd(l,xf)
                for k,_ in groups:
                    a=store[('q',k)]; b=store[('f',k)]
                    H[k]=H.get(k,0)+2*a.T@a; Cx[k]=Cx.get(k,0)+2*b.T@a; nsm[k]=nsm.get(k,0)+a.shape[0]
        for h in hds: h.remove()
        return mods,{k:H[k]/nsm[k] for k in H},{k:Cx[k]/nsm[k] for k in Cx}
    def signs(oli,key,C): return (torch.randint(0,2,(C,),generator=torch.Generator().manual_seed(1000*oli+len(key)),dtype=torch.float32)*2-1).cuda()
    LS=[2,3,4,5,6,8,12,16]
    # ---------- pass A: sensitivity tables (fp prefix) + block influence
    spath=f"{HFD}/dense/sens_{tag}_g{g}_dz{dz}.json"
    if os.path.exists(spath): S=json.load(open(spath))
    else:
        S=dict(tab={},bi=[]); cur=[x.clone() for x in inps]
        for oli,l in enumerate(layers):
            l.cuda(); mods,Hs=collect(l,cur)
            with torch.no_grad():
                nxt=[fwd(l,x) for x in cur]
                S["bi"].append(float(np.mean([1-torch.nn.functional.cosine_similarity(a.float().flatten(0,1),b.float().flatten(0,1),dim=-1).mean().item() for a,b in zip(cur[:16],nxt[:16])])))
                for key,mem in groups:
                    H=Hs[key]; C=H.shape[0]; sg=signs(oli,key,C); Hr=rot(rot(H,sg).T.contiguous(),sg).T.contiguous()
                    for n in mem:
                        W=rot(mods[n].weight.data.float(),sg); ref=float(((W@Hr)*W).sum()); tab={}
                        for Lv in LS:
                            Q,h=rtn(W,Lv); Dd=W-Q; tab[Lv]=[float(((Dd@Hr)*Dd).sum())/ref,ent(h)]
                        S["tab"][f"{oli}.{n}"]=dict(n=W.numel(),tab=tab)
            cur=nxt; (None if GPUR else l.cpu()); del Hs; torch.cuda.empty_cache()
            if oli%8==0: print("sensA",oli,round(time.time()-t0),flush=True)
        json.dump(S,open(spath,"w")); (vol.commit() if HFD=='/hf' else None); S=json.loads(json.dumps(S))
    if mode=="sens": print("SENS_DONE",tag,round(time.time()-t0),flush=True); return dict(tag=tag,secs=time.time()-t0)
    # ---------- layer drop
    nd=cfg.get("drop",0); bi=S["bi"]
    dropl=sorted(sorted(range(2,nl0-1),key=lambda i:bi[i])[:nd]) if nd else []
    keep=[i for i in range(nl0) if i not in dropl]
    qs=cfg.get('qstride',1); qo=cfg.get('qoff',0)
    qkeep=[i for i in keep if i%qs==qo%qs] if not cfg.get('qlist') else [i for i in keep if i in cfg['qlist']]
    if cfg.get('qlist') or qs>1: qs=max(qs,2)
    # ---------- fixed bytes
    hid=model.config.hidden_size; V=model.get_input_embeddings().weight.shape[0]
    tied=model.config.tie_word_embeddings
    lin_total=sum(S["tab"][f"{i}.{n}"]["n"] for i in keep for n in linn)
    other=sum(p.numel() for i in keep for p in layers[i].parameters())-lin_total+model.model.norm.weight.numel()
    info=HfApi().model_info(M,files_metadata=True); fb=sum(s.size for s in info.siblings if s.rfilename in("tokenizer.json","tokenizer_config.json","config.json","generation_config.json"))
    CODE=150000; fixed_bytes=other*2+fb+CODE
    eL=cfg.get("emb_L",16); hL=cfg.get("head_L",16); ge=cfg.get("g_emb",128)
    # emb/head entropy estimates by RTN
    gsave=g
    def est_ent(W,Lv):
        sg=torch.ones(W.shape[1]).cuda()
        gen=torch.Generator().manual_seed(7); idx=torch.randperm(W.shape[0],generator=gen)[:4096]
        Wr=rot(W[idx].cuda().float(),sg); Q,h=rtn(Wr,Lv) if Lv<999 else (Wr,None); return ent(h)
    # (g must equal for rtn; use g for head, emb uses ge via temporary swap below)
    ent_e=est_ent(model.get_input_embeddings().weight.data,eL) if eL<999 else 16.0
    ent_h=est_ent(model.get_output_embeddings().weight.data,hL) if hL<999 else 16.0
    emb_bytes=V*hid*(ent_e*1.005+GB/g)/8 if eL<999 else V*hid*2
    head_bytes=V*hid*(ent_h*1.005+GB/g)/8 if (hL<999 and not tied) else (0 if tied else V*hid*2)
    def kl_tables():
        kp=f"{HFD}/dense/klsens_{tag}_g{g}_{cfg.get('klcal','t1')}.json"
        if os.path.exists(kp): return json.load(open(kp))
        model.cuda(); KS={}; nk=cfg.get("nkl",8); sq=stream[:nk,:].cuda()
        gmap={n:k for k,mem in groups for n in mem}
        with torch.no_grad():
            blp=[torch.log_softmax(model(input_ids=sq[i:i+1]).logits.float(),-1) for i in range(nk)]
            def klnow():
                t=0.0
                for i in range(nk):
                    lq=torch.log_softmax(model(input_ids=sq[i:i+1]).logits.float(),-1); t+=float((blp[i].exp()*(blp[i]-lq)).sum(-1).mean())
                return t/nk
            for li in keep:
                for n in linn:
                    m=model.model.layers[li].get_submodule(n); W0=m.weight.data.clone(); C=W0.shape[1]; sg=signs(li,gmap[n],C); Wr=rot(W0.float(),sg); KS[f"{li}.{n}"]={}
                    for Lv in [2,3,4,8]:
                        Q,h=rtn(Wr,Lv); m.weight.data=unrot(Q,sg).to(torch.bfloat16); KS[f"{li}.{n}"][str(Lv)]=klnow()
                    m.weight.data=W0
                if li%8==0: print("klsens",li,round(time.time()-t0),flush=True)
        del blp; (None if GPUR else model.cpu()); torch.cuda.empty_cache()
        json.dump(KS,open(kp,"w")); return KS
    # ---------- allocation
    w=np.array(bi); gam=cfg.get("gamma",0.0)
    lw={i:(w[i]/w.mean())**gam for i in keep}
    levels={}
    if cfg.get("L"):
        for i in keep:
            for n in linn: levels[f"{i}.{n}"]=cfg["L"]
    else:
        target=cfg["target_ratio"]*base_bytes; body_bits=(target-fixed_bytes-emb_bytes-head_bytes)*8
        items=[(f"{i}.{n}",S["tab"][f"{i}.{n}"],lw[i]) for i in keep for n in linn]
        if cfg.get("alloc")=="kl":
            KS=kl_tables(); KLS=[2,3,4,8]
            items=[(k,dict(n=it['n'],tab={str(Lv):[KS[k][str(Lv)],it['tab'][str(Lv)][1]] for Lv in KLS}),1.0) for k,it,_ in items]
            LSX=KLS
        def pick(lam):
            tot=0;ch={}
            for k,it,wt in items:
                best=None
                for Lv in (LSX if cfg.get('alloc')=='kl' else LS):
                    e,H_=it["tab"][str(Lv)]; b=it["n"]*(H_*1.005+GB/g); c=(wt*e*1e3 if cfg.get('alloc')=='kl' else wt*e*it["n"]/lin_total*1e3)+lam*b
                    if best is None or c<best[0]: best=(c,Lv,b)
                ch[k]=best[1]; tot+=best[2]
            return tot,ch
        lo,hi=1e-14,1e-3
        for _ in range(60):
            mid=math.sqrt(lo*hi); tot,ch=pick(mid)
            if tot>body_bits: lo=mid
            else: hi=mid
        tot,levels=pick(hi); print("alloc body_bits/param",tot/lin_total,"target",body_bits/lin_total,flush=True)
    # ---------- pass B: GPTQ, quantized prefix
    EP=bool(cfg.get('ep')); curf=[x.clone() for x in inps] if EP else None
    cur=[x.clone() for x in inps]; qbits=0.0; hist_by_L=collections=None
    from collections import Counter
    Lcount=Counter(); layer_bpw=[]
    for oli in keep:
        l=layers[oli]; l.cuda()
        if oli not in qkeep:
            with torch.no_grad(): cur=[fwd(l,x) for x in cur]
            (None if GPUR else l.cpu()); continue
        if EP:
            mods,Hs,Cs=collect2(l,cur,curf)
            with torch.no_grad(): curf_next=[fwd(l,x) for x in curf]
        else: mods,Hs=collect(l,cur)
        lb=0.0; ln=0
        for key,mem in groups:
            H=Hs[key]; C=H.shape[0]; sg=signs(oli,key,C); Hr=rot(rot(H,sg).T.contiguous(),sg).T.contiguous()
            if EP:
                Cr=rot(rot(Cs[key],sg).T.contiguous(),sg).T.contiguous(); lam=cfg.get('ep_lam',0.01)*torch.mean(torch.diag(Hr))
                Hreg=Hr+lam*torch.eye(C,device=Hr.device)
            for n in mem:
                Lv=levels[f"{oli}.{n}"]; Wo=mods[n].weight.data; W=rot(Wo.float(),sg)
                if EP: W=(W+cfg.get('ep_alpha',1.0)*torch.linalg.solve(Hreg,(W@(Cr-Hr)).T).T).contiguous()
                Q,h=gptq(W,Hr,Lv); mods[n].weight.data=unrot(Q,sg).to(torch.bfloat16)
                nn_=W.numel(); b=nn_*(ent(h)*1.005+GB/g); qbits+=b; lb+=b; ln+=nn_; Lcount[Lv]+=nn_
        layer_bpw.append(lb/ln)
        with torch.no_grad(): cur=[fwd(l,x) for x in cur]
        if EP: curf=curf_next
        del Hs; (None if GPUR else l.cpu()); torch.cuda.empty_cache()
        if oli%8==0: print("passB",oli,round(time.time()-t0),flush=True)
    # final hidden for head hessian
    Hh=0;nh=0
    with torch.no_grad():
        nm=model.model.norm.cuda()
        for x in cur:
            y=nm(x).reshape(-1,hid).float(); Hh=Hh+2*y.T@y; nh+=y.shape[0]
    Hh=Hh/nh; (None if GPUR else nm.cpu())
    if dropl:
        model.model.layers=nn.ModuleList([layers[i] for i in keep]); model.config.num_hidden_layers=len(keep)
        for ni,l in enumerate(model.model.layers): l.self_attn.layer_idx=ni
    # head
    head=model.get_output_embeddings()
    if qs>1: hL=eL=999
    if hL<999:
        sg=signs(99999,"head",hid); Wh=rot(head.weight.data.cuda().float(),sg); Hr=rot(rot(Hh,sg).T.contiguous(),sg).T.contiguous()
        Q,h=gptq(Wh,Hr,hL); head_bits=head.weight.numel()*(ent(h)*1.005+GB/g); head.weight.data=unrot(Q,sg).to(torch.bfloat16).to('cuda' if GPUR else 'cpu'); del Q,Wh
    else: head_bits=head.weight.numel()*16
    if tied: head_bits=0
    emb=model.get_input_embeddings()
    if eL<999:
        sg=signs(88888,"emb",hid); We=rot(emb.weight.data.cuda().float(),sg); g_s=g; 
        # embedding uses group ge: rtn() reads closure g, so temporarily emulate by reshaping
        Wq=torch.empty_like(We); hist=torch.zeros(eL,device="cuda")
        for r in range(0,We.shape[0],8192):
            x=We[r:r+8192].reshape(-1,ge); s=qparams(x,eL); q,qi=quant(x,s,eL,dz); Wq[r:r+8192]=q.view(-1,hid); hist+=torch.bincount(qi.long().flatten(),minlength=eL).float()
        emb_bits=emb.weight.numel()*(ent(hist)*1.005+GB/ge); emb.weight.data=unrot(Wq,sg).to(torch.bfloat16).to('cuda' if GPUR else 'cpu')
        if tied: head.weight=emb.weight
    else: emb_bits=emb.weight.numel()*16
    tot_bytes=qbits/8+head_bits/8+emb_bits/8+fixed_bytes
    del inps,cur; torch.cuda.empty_cache()
    model.cuda()
    if cfg.get('t1'):
        run_t1(); return t1_result(locals_=dict(tot_bytes=tot_bytes,base_bytes=base_bytes,qbits=qbits,head_bits=head_bits,emb_bits=emb_bits,fixed_bytes=fixed_bytes,levels=levels,dropped=dropl,lin_total=lin_total))
    # ---------- eval
    nk=cfg.get('nev',0)
    def subidx(ev): 
        km=kind_masks(ev); return sorted(i for v in km.values() for i in (v[:nk] if nk else v))
    sid=subidx(ev_id); sod=subidx(ev_ood)
    G_id=gen_all([ev_id[i]["prompt"] for i in sid]); G_od=gen_all([ev_ood[i]["prompt"] for i in sod])
    B_id=[base["gid"][i][:NEW] for i in sid]; B_od=[base["god"][i][:NEW] for i in sod]
    ev_id=[ev_id[i] for i in sid]; ev_ood=[ev_ood[i] for i in sod]
    def agree(G,B,ev):
        o={}; 
        for k,idx in kind_masks(ev).items():
            o[k]=dict(exact64=float(np.mean([G[i]==B[i] for i in idx])),exact16=float(np.mean([G[i][:16]==B[i][:16] for i in idx])),
                      prefix_frac=float(np.mean([(next((j for j,(a,b) in enumerate(zip(G[i],B[i])) if a!=b),min(len(G[i]),len(B[i]))+(0 if len(G[i])==len(B[i]) else 0))/max(1,len(B[i]))) for i in idx])))
        allidx=range(len(ev)); o["ALL"]=dict(exact64=float(np.mean([G[i]==B[i] for i in allidx])),exact16=float(np.mean([G[i][:16]==B[i][:16] for i in allidx])),
              prefix_frac=float(np.mean([o[k]["prefix_frac"]*len(kind_masks(ev)[k]) for k in kind_masks(ev)]))/len(ev)*1.0 if False else float(np.mean([next((j for j,(a,b) in enumerate(zip(G[i],B[i])) if a!=b),len(B[i]))/max(1,len(B[i])) for i in allidx])))
        return o
    def tfmet(items,btf):
        c=tf_logp(items); t1=[];kl=[]
        for (ci,cl),(bi_,bl) in zip(c,btf):
            bi_=np.array(bi_);bl=np.array(bl)
            t1+=list(ci[:,0]==bi_[:,0])
            # cand logprob at base top-K ids: recompute needs full dist -> done below via lookup
        return c
    # candidate logprobs at base ids: recompute with direct gather
    @torch.no_grad()
    def tf_cmp(items,btf):
        t1=[];kl=[]
        for (pi,ri),(bids,blp) in zip(items,btf):
            bids=torch.tensor(bids).cuda(); blp=torch.tensor(blp).cuda(); P=len(ri)
            lg=model(input_ids=torch.tensor([(pi+ri)[-640:]]).cuda()).logits[0,-P-1:-1].float(); lp=torch.log_softmax(lg,-1)
            t1+=(lp.argmax(-1)==bids[:,0]).cpu().tolist()
            cl=lp.gather(1,bids); pb=blp.exp(); tb=(1-pb.sum(1)).clamp(min=1e-9); tc=(1-cl.exp().sum(1)).clamp(min=1e-9)
            k=(pb*(blp-cl)).sum(1)+tb*(tb.log()-tc.log()); kl+=k.cpu().tolist()
        return float(np.mean(t1)),float(np.mean(kl))
    ntf=cfg.get('ntf',0) or len(tfs); ti=list(range(0,len(tfs),max(1,len(tfs)//ntf)))[:ntf]
    allt=tf_items(); tf_id=tf_cmp([allt[i] for i in ti],[base["tf"][i] for i in ti]); sel=base["ood_sel"]
    ood_full=D["eval_ood"]; ts=list(range(0,len(sel),max(1,len(sel)//max(1,ntf//3))))[:max(1,ntf//3)] if cfg.get('ntf') else list(range(len(sel)))
    tf_od=tf_cmp([(enc_prompt(ood_full[sel[j]]["prompt"]),base["god"][sel[j]]) for j in ts],[base["tfo"][j] for j in ts])
    A_id=agree(G_id,B_id,ev_id); A_od=agree(G_od,B_od,ev_ood)
    res=dict(cfg=cfg,model=M,bytes=tot_bytes,base_bytes=base_bytes,ratio=tot_bytes/base_bytes,
      parts_MB=dict(body=qbits/8e6,emb=emb_bits/8e6,head=head_bits/8e6,norms_fixed=other*2/1e6,files=fb/1e6,code=CODE/1e6),
      body_bpw=qbits/lin_total,L_hist={str(k):v/lin_total for k,v in Lcount.items()},dropped=dropl,
      id=dict(gen=A_id,tf_top1=tf_id[0],tf_kl=tf_id[1]),ood=dict(gen=A_od,tf_top1=tf_od[0],tf_kl=tf_od[1]),
      base_noise_floor=base["floor_bs32_vs_bs8"],secs=time.time()-t0,gens=dict(id=G_id,ood=G_od,sid=sid,sod=sod))
    print("RESULT_JSON "+json.dumps(res,default=str),flush=True)
    json.dump(res,open(f"{HFD}/dense/res_{tag}_{name}.json","w"),default=str); (vol.commit() if HFD=='/hf' else None); return res

fn={}
for k,v in SIZES.items():
    fn[k]=app.function(image=img,timeout=6*3600,volumes={"/hf":vol},name="run_"+k,serialized=False,**v)(_run) if False else None
run_s=app.function(image=img,timeout=6*3600,volumes={"/hf":vol},name="run_s",**SIZES["s"])(_run)
run_m=app.function(image=img,timeout=6*3600,volumes={"/hf":vol},name="run_m",**SIZES["m"])(_run)
run_x=app.function(image=img,timeout=6*3600,volumes={"/hf":vol},name="run_x",gpu=["L40S","A100-40GB"],memory=100000)(_run)
run_l=app.function(image=img,timeout=6*3600,volumes={"/hf":vol},name="run_l",**SIZES["l"])(_run)

@app.local_entrypoint()
def main(cfgs:str,out:str):
    import time
    cs=json.load(open(cfgs)); hs=[]
    for c in cs:
        f={"s":run_s,"m":run_m,"l":run_l}[c.get("size","s")]
        hs.append(f.spawn(c)); time.sleep(10)
    rs=[]
    for h in hs:
        try: rs.append(h.get())
        except Exception as e: rs.append({"error":repr(e)})
        json.dump(rs,open(out,"w"),default=str)
