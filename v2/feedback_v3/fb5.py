"""feedback_v3 round 3 (fb5): natural-mistake + new-tool T1 scenarios, controls (empty/mixed/random-tool), >=10 annotation draws, real-codec stand-in. (head copied from fb4)
(fb4 doc:: task B rerun with proper swap control, T1 real tool-use targets (S1-dev/S3-dev only),
lm_head codec survival, widened lmhead_lda alpha/ridge sweep, lmhead_lda + late-layer down_proj blend.
modal run --detach fb4.py --model Qwen/Qwen3-8B --scen T1pairS:first,T1pairP:first,T1new:first,T1pairS:last,T1new:last
scenarios: A, B (proxy data cached by fb3 on the volume), T1pairS (alias name = G+'_v2', distinguishing token LATER), T1pairP (alias = 'v2_'+G, FIRST token),
T1new (nocall queries -> call to NEW tool escalate_to_specialist not in the tool list). fmt first = native {"name":..,"arguments":..}; last = {"arguments":..,"name":..}.
Fitted-to-data quantities: see README (generic covariance Sigma and mean mu from unlabeled text; alpha, ridge chosen on dev; class means from the <=25 annotations)."""
import modal, os, json
HERE=os.path.dirname(os.path.abspath(__file__)); T1=os.path.expanduser("~/ten-percent/v2/tournament/T1")
app=modal.App("tenpct-fb5")
img=(modal.Image.debian_slim(python_version="3.11").pip_install("torch==2.5.1","transformers==4.56.2","accelerate","numpy","datasets")
     .add_local_file(f"{T1}/scorer.py","/app/scorer.py").add_local_file(f"{T1}/data/s1_dev.jsonl","/app/s1_dev.jsonl")
     .add_local_file(f"{T1}/data/s3_dev.jsonl","/app/s3_dev.jsonl").add_local_file(f"{T1}/data/s1_calib.jsonl","/app/s1_calib.jsonl"))
vol=modal.Volume.from_name("tenpct-v2-hf")
GPU=os.environ.get("FB4_GPU","A100-80GB")

@app.function(image=img,gpu=GPU,timeout=36000,volumes={"/vol":vol})
def run(model="Qwen/Qwen3-8B",scen="nat",ptag="",ndraw=10,only=""):
    import torch,numpy as np,random,hashlib,re,time,sys
    VC=(lambda: None) if os.environ.get('NOVOL') else vol.commit
    sys.path.insert(0,"/app"); from scorer import score_item
    os.environ["HF_HOME"]="/vol/hf"; os.makedirs("/vol/fb3",exist_ok=True)
    from transformers import AutoTokenizer,AutoModelForCausalLM
    T0=time.time(); log=lambda *a: print(f"[{time.time()-T0:6.0f}s]",*a,flush=True)
    tok=AutoTokenizer.from_pretrained(model); tok.padding_side="left"
    lm=AutoModelForCausalLM.from_pretrained(model,dtype=torch.bfloat16).cuda().eval()
    for p in lm.parameters(): p.requires_grad_(False)
    dev="cuda"; NL=lm.config.num_hidden_layers; layers_all=lm.model.layers; D_h=lm.config.hidden_size
    mname=model.split("/")[-1]
    h8=lambda s:int(hashlib.sha256(s.encode()).hexdigest()[:8],16)
    def chat(u,tools=None,fmt="first"):
        kw=dict(tokenize=False,add_generation_prompt=True)
        try: p=tok.apply_chat_template([{"role":"user","content":u}],tools=tools,enable_thinking=False,**kw)
        except TypeError: p=tok.apply_chat_template([{"role":"user","content":u}],tools=tools,**kw)
        if fmt=="last": p=p.replace('{"name": <function-name>, "arguments": <args-json-object>}','{"arguments": <args-json-object>, "name": <function-name>}')
        return p
    BS={"v":48}
    @torch.no_grad()
    def gen(prompts,mx,bs=None,skip=True):
        bs=bs or BS["v"]; out=[None]*len(prompts); order=sorted(range(len(prompts)),key=lambda i:len(prompts[i]))
        for s in range(0,len(order),bs):
            idx=order[s:s+bs]; enc=tok([prompts[i] for i in idx],return_tensors="pt",padding=True).to(dev)
            o=lm.generate(**enc,max_new_tokens=mx,do_sample=False,pad_token_id=tok.pad_token_id)
            for j,i in enumerate(idx): out[i]=o[j,enc.input_ids.shape[1]:].tolist()
        return out
    dec=lambda ids: tok.decode(ids,skip_special_tokens=False).replace("<|im_end|>","").replace("<|endoftext|>","").strip()
    GEN=["Explain why the sky is blue in two sentences.","What is the capital of Australia?","Write a haiku about autumn.","Translate 'good morning, how are you?' into French.","Give three tips for a job interview.","What is 17 times 23?","Summarize the plot of Romeo and Juliet in one sentence.","Write a Python function that reverses a string.","What causes tides?","Define photosynthesis briefly.","Who wrote Pride and Prejudice?","How do I boil an egg?","Explain what an API is to a beginner.","Name three renewable energy sources.","What is the difference between weather and climate?","Write a polite email declining a meeting.","Describe the water cycle.","What does HTTP stand for?","Why do we need sleep?","Convert 100 Fahrenheit to Celsius.","What is machine learning?","How does a binary search work?","What is inflation?","Explain recursion with an example.","What year did World War II end?","Write a SQL query to count rows in a table named users.","Fix the grammar: 'she dont like apples'.","What is DNA?","Explain supply and demand.","How do vaccines work?"]
    GP=[chat(g) for g in GEN]; base_gen=gen(GP,32)
    def drift():
        o=gen(GP,32); return dict(identical=float(np.mean([a==b for a,b in zip(o,base_gen)])),token_agree=float(np.mean([np.mean([x==y for x,y in zip(a[:32],b[:32])]) if min(len(a),len(b))>0 else 1.0 for a,b in zip(o,base_gen)])))
    # ---------------- generic stats (unlabeled text only): wikitext + chat prompts + (T1) unlabeled s1_calib prompts rendered with the tool template
    cov_texts=[]
    try:
        from datasets import load_dataset
        wt=load_dataset("wikitext","wikitext-103-raw-v1",split="train[:3000]"); buf=""
        for r in wt:
            if r["text"].strip() and not r["text"].startswith(" ="): buf+=r["text"].strip()+"\n"
            if len(buf)>3000: cov_texts.append(buf);buf=""
            if len(cov_texts)>=60: break
    except Exception as e: log("wikitext fail",e)
    chat_cov=[g+tok.decode(o,skip_special_tokens=False) for g,o in zip(GP,base_gen)]
    calib=[json.loads(l) for l in open("/app/s1_calib.jsonl")]; random.Random(0).shuffle(calib)
    t1_cov=[chat(c["query"],[{"type":"function","function":t} for t in c["tools"]]) for c in calib[:120] if len(c["tools"])<=6]
    COV={}; MASKH=[None]
    mod_of=lambda l,kind: layers_all[l].mlp.down_proj if kind=="down" else layers_all[l].self_attn.o_proj
    @torch.no_grad()
    def get_cov(ls,kind,extra):
        key=(tuple(ls),kind,extra)
        if key in COV: return COV[key]
        acc={l:None for l in ls}; cnt=[0]
        def mk(l):
            def f(m,inp):
                x=inp[0].float().reshape(-1,inp[0].shape[-1])[MASKH[0].reshape(-1)]
                acc[l]=x.T@x if acc[l] is None else acc[l]+x.T@x
                if l==ls[0]: cnt[0]+=x.shape[0]
            return f
        hs=[mod_of(l,kind).register_forward_pre_hook(mk(l)) for l in ls]
        texts=cov_texts+chat_cov*3+(t1_cov if extra else [])
        for s in range(0,len(texts),4):
            enc=tok(texts[s:s+4],return_tensors="pt",padding=True,truncation=True,max_length=1024).to(dev); MASKH[0]=enc.attention_mask.bool(); lm.model(**enc)
        for h in hs: h.remove()
        COV[key]={l:acc[l]/cnt[0] for l in ls}; return COV[key]
    @torch.no_grad()
    def gstats(extra):
        xs=[]; texts=chat_cov+(t1_cov if extra else [])
        for s in range(0,len(texts),4):
            enc=tok(texts[s:s+4],return_tensors="pt",padding=True,truncation=True,max_length=1024).to(dev)
            fh={}; h=lm.model.norm.register_forward_hook(lambda m,i_,o: fh.__setitem__("h",o)); lm.model(**enc); h.remove()
            xs.append(fh["h"].float()[enc.attention_mask.bool()])
        X=torch.cat(xs); return X.mean(0),torch.cov(X.T)
    final_norm_w=lm.model.norm.weight.float(); Wu=lm.lm_head.weight
    saved={}
    def save(mod):
        if id(mod.weight) not in saved: saved[id(mod.weight)]=(mod.weight,mod.weight.data.clone())
    def restore():
        for w,d in saved.values(): w.data.copy_(d)
        saved.clear()
    # ---------------- edit machinery (same as fb3)
    @torch.no_grad()
    def sites(anns,sc_mx=None):
        S=[]
        for it in anns:
            p=tok(it["prompt"],add_special_tokens=False).input_ids; g=tok(it["target"],add_special_tokens=False).input_ids
            ids=torch.tensor([p+g],device=dev); lg=lm(ids).logits[0].float()
            for j,t in enumerate(g):
                pos=len(p)+j-1; am=int(lg[pos].argmax())
                if am!=t: S.append(dict(ids=ids[0],pos=pos,tgt=t,wrong=am))
        return S
    @torch.no_grad()
    def capture(S,ls,kind,out_layer):
        K={l:[] for l in ls}; H=[]
        for s in S:
            ids=s["ids"][None]; ks={}
            hs=[mod_of(l,kind).register_forward_pre_hook(lambda m,inp,l=l: ks.__setitem__(l,inp[0][0,s["pos"]].float())) for l in ls]
            ho={}; h2=layers_all[out_layer].register_forward_hook(lambda m,i_,o: ho.__setitem__("h",(o[0] if isinstance(o,tuple) else o)[0,s["pos"]].float()))
            lm.model(ids)
            for h in hs: h.remove()
            h2.remove()
            for l in ls: K[l].append(ks[l])
            H.append(ho["h"])
        return {l:torch.stack(K[l]) for l in ls},torch.stack(H)
    CGRID=[0.05,0.1,0.2,0.35,0.5,0.75,1,1.5,2,3,4,6]
    @torch.no_grad()
    def targets(S,out_layer,margin):
        Z=[]
        for s in S:
            ids=s["ids"][None]; u=final_norm_w*(Wu[s["tgt"]].float()-Wu[s["wrong"]].float()); u=u/u.norm(); best=None
            for c in CGRID:
                ho={}
                def hk(m,i_,o,c=c):
                    h=(o[0] if isinstance(o,tuple) else o).clone(); hn=h[0,s["pos"]].float().norm()
                    h[0,s["pos"]]=(h[0,s["pos"]].float()+c*hn*u).to(h.dtype); ho["d"]=c*hn*u
                    return (h,)+tuple(o[1:]) if isinstance(o,tuple) else h
                hh=layers_all[out_layer].register_forward_hook(hk); lg=lm(ids).logits[0,s["pos"]].float(); hh.remove(); best=ho["d"]
                if int(lg.argmax())==s["tgt"] and (lg[s["tgt"]]-lg[s["wrong"]])>=margin: break
            Z.append(best)
        return torch.stack(Z)
    EXTRA=[False]
    @torch.no_grad()
    def memit(S,ls,kind,lam,margin=2.0):
        out_layer=ls[-1]; Z=targets(S,out_layer,margin); _,H0=capture(S,ls[-1:],kind,out_layer); target=H0+Z; C=get_cov(ls,kind,EXTRA[0])
        for li,l in enumerate(ls):
            K,H=capture(S,[l],kind,out_layer); K=K[l]; R=(target-H)/(len(ls)-li)
            A=K.T@K+lam*C[l]+1e-4*torch.eye(K.shape[1],device=dev)*C[l].diagonal().mean()
            dW=torch.linalg.solve(A,K.T@R).T; m=mod_of(l,kind); save(m); m.weight.data.copy_((m.weight.data.float()+dW).to(m.weight.dtype))
    G={}
    @torch.no_grad()
    def lmhead(S,alpha,mode,ridge=1e-2):
        hs=[]
        for s in S:
            fh={}; h=lm.model.norm.register_forward_hook(lambda m,i_,o: fh.__setitem__("h",o[0,s["pos"]].float())); lm.model(s["ids"][None]); h.remove(); hs.append(fh["h"])
        Hs=torch.stack(hs); by={}
        for i,s in enumerate(S): by.setdefault(s["tgt"],[]).append(i)
        save(lm.lm_head); edited=[]
        for t,ix in by.items():
            d=Hs[ix].mean(0) if mode=="plain" else Hs[ix].mean(0)-G["mu"]
            if mode=="lda": d=torch.linalg.solve(G["sig"]+ridge*G["sig"].diagonal().mean()*torch.eye(D_h,device=dev),d)
            d=d/d.norm(); w=lm.lm_head.weight.data
            w[t]=(w[t].float()+alpha*w[t].float().norm()*d).to(w.dtype); edited.append(t)
        return edited
    # 4-bit group-128 RTN for the lm_head codec test
    def q4rows(W,g=128):
        Wg=W.float().reshape(W.shape[0],-1,g); mn=Wg.amin(-1,keepdim=True); mx=Wg.amax(-1,keepdim=True); sc=(mx-mn).clamp_min(1e-8)/15
        return sc,mn
    def q4apply(W,sc,mn,g=128):
        Wg=W.float().reshape(W.shape[0],-1,g); return (((Wg-mn)/sc).round().clamp(0,15)*sc+mn).reshape(W.shape)

    # ================= fb5 tail =================
    from scorer import parse_calls
    VCm=VC
    def lmhead_cal(S,ridge=0.1,margin=1.0,amax=2.0,frac=0.9,pos=False):
        """lm_head LDA edit with SELF-CALIBRATED per-token alpha: smallest alpha in a grid that makes the annotated sites' top-1 flip with margin
        (forward-free: only the edited row's logit changes, so flip test is exact from stored final-normed hiddens). No dev tuning of alpha."""
        hs=[]
        for s in S:
            fh={}; h=lm.model.norm.register_forward_hook(lambda m,i_,o: fh.__setitem__("h",o[0,s["pos"]].float())); lm.model(s["ids"][None]); h.remove(); hs.append(fh["h"])
        Hs=torch.stack(hs); by={}
        for i,s in enumerate(S): by.setdefault(s["tgt"],[]).append(i)
        save(lm.lm_head); W=lm.lm_head.weight.data; base_logits=Hs@W.float().T; edited=[]; alphas={}
        for t,ix in by.items():
            mu_,sg_=(G["mu_pos"],G["sig_pos"]) if pos else (G["mu"],G["sig"])
            d=Hs[ix].mean(0)-mu_; d=torch.linalg.solve(sg_+ridge*sg_.diagonal().mean()*torch.eye(D_h,device=dev),d); d=d/d.norm()
            wt=W[t].float(); other=base_logits[ix].clone(); other[:,t]=-1e9; mx=other.max(1).values; chosen=amax
            for a in [0.05,0.1,0.15,0.2,0.3,0.45,0.7,1.0,1.5,2.0]:
                newl=Hs[ix]@(wt+a*wt.norm()*d)
                if float(((newl-mx)>=margin).float().mean())>=frac: chosen=a;break
            W[t]=(wt+chosen*wt.norm()*d).to(W.dtype); edited.append(t); alphas[int(t)]=chosen
        return edited
    # ---- real-codec stand-in: sign-diag x block-Hadamard(128) rotation, L-level symmetric grid, per-128 group fp16 scale (from compress_A/sweep_A.py), ideal-rANS bytes = entropy of level histogram
    import math
    PGRID=torch.tensor([1.0,0.9,0.8,0.7,0.62,0.55,0.5,0.45,0.4,0.36,0.32,0.28,0.25],device=dev); HB=128
    def hada(n):
        Hm=torch.ones(1,1)
        while Hm.shape[0]<n: Hm=torch.cat([torch.cat([Hm,Hm],1),torch.cat([Hm,-Hm],1)],0)
        return (Hm/math.sqrt(n)).to(dev)
    Hb=hada(HB)
    def qpar(x,Lv):
        half=(Lv-1)/2; am=x.abs().max(1,keepdim=True)[0].clamp(min=1e-8); best=None
        for p in PGRID:
            s=(am*p/half).half().float().clamp(min=1e-8); q=(torch.clamp(torch.round(x/s+half),0,Lv-1)-half)*s; e=((q-x)**2).sum(1,keepdim=True)
            if best is None: best=(e,s)
            else: m=e<best[0]; best=(torch.where(m,e,best[0]),torch.where(m,s,best[1]))
        return best[1]
    def rot(M,sg): R,C=M.shape; return ((M*sg).view(R,C//HB,HB)@Hb).reshape(R,C)
    def unrot(M,sg): R,C=M.shape; return ((M.view(R,C//HB,HB)@Hb).reshape(R,C))*sg
    def enc_rows(Wrows,sg,Lv,scale=None):
        """returns dequantised rows (original domain), level histogram, scales (m*ngroups,1)"""
        R,C=Wrows.shape; X=rot(Wrows.float(),sg).reshape(-1,128); s=qpar(X,Lv) if scale is None else scale
        half=(Lv-1)/2; qi=torch.clamp(torch.round(X/s+half),0,Lv-1); Q=((qi-half)*s).reshape(R,C)
        return unrot(Q,sg),torch.bincount(qi.long().flatten(),minlength=Lv).double(),s
    ent_bytes=lambda hist,nscale: float((-(hist/hist.sum()*(hist/hist.sum()).clamp_min(1e-30).log2()).sum())*hist.sum()/8+2*nscale)
    # ---- data
    mname_=mname; raw=[json.loads(l) for f in ("/app/s1_dev.jsonl","/app/s3_dev.jsonl") for l in open(f)]
    singles=[x for x in raw if x["kind"]=="single" and len(x["gold"])==1 and len(x["tools"])<=6]
    nocall=[x for x in raw if x["kind"]=="nocall" and len(x["tools"])<=6]
    ALLTOOLS=sorted({t["name"] for x in singles for t in x["tools"]})
    gtxt=lambda calls: "\n".join("<tool_call>\n"+json.dumps({"name":c["name"],"arguments":c["arguments"]},ensure_ascii=False)+"\n</tool_call>" for c in calls)
    def mk(x,gold,tools=None,wrong=None):
        tools=tools or x["tools"]; it=dict(x,gold=gold,tools=tools)
        return dict(prompt=chat(x["query"],[{"type":"function","function":t} for t in tools]),gold=gold,item=it,target=gtxt(gold)+"<|im_end|>",wrong=wrong,id=x["id"],tool=gold[0]["name"])
    def sc(it,o): return bool(score_item(dec(o),it["item"])["correct"])
    BS["v"]=24
    def ev(L,ret=False):
        if not L: return (np.zeros(0,int),[]) if ret else np.zeros(0,int)
        o=gen([it["prompt"] for it in L],100); r=np.array([int(sc(it,x)) for it,x in zip(L,o)])
        return (r,[dec(x) for x in o]) if ret else r
    # ---- base eval of ALL singles (cached) -> natural mistakes
    cf=f"/vol/fb3/base_{mname}_singles.json"
    if os.path.exists(cf): BASE=json.load(open(cf))
    else:
        allit=[mk(x,x["gold"]) for x in singles]; o=gen([it["prompt"] for it in allit],100); BASE={}
        for it,x in zip(allit,o): t=dec(x); BASE[it["id"]]=dict(text=t,correct=bool(sc(it,x)))
        json.dump(BASE,open(cf,"w")); VC(); log("base singles eval done",sum(v["correct"] for v in BASE.values()),"/",len(BASE))
    def classify(x,text):
        calls,fail=parse_calls(text); g=x["gold"][0]
        if fail: return "parsefail"
        if len(calls)==0: return "abstain"
        if len(calls)>1: return "multi"
        n,a=calls[0]; names={t["name"] for t in x["tools"]}
        if n!=g["name"]: return "sibling" if (n in names and len(names)>=2) else "wrongname"
        ga=g["arguments"]; ks=set(a)|set(ga)
        if set(a)!=set(ga): return "argkeys"
        diffs=[k for k in ga if a.get(k)!=ga[k]]
        if not diffs: return "argother"
        k=diffs[0]; pv,gv=a[k],ga[k]
        if type(pv)!=type(gv) and not (isinstance(pv,(int,float)) and isinstance(gv,(int,float))): return "argtype"
        if isinstance(pv,str) and isinstance(gv,str):
            return "argcase" if pv.strip().lower()==gv.strip().lower() else "argstr"
        return "argnum"
    byid={x["id"]:x for x in singles}; FAIL={}
    for i,v in BASE.items():
        if not v["correct"]: FAIL.setdefault(classify(byid[i],v["text"]),[]).append(i)
    cnt={c:(len(v),len({byid[i]["gold"][0]["name"] for i in v})) for c,v in FAIL.items()}; log("failure classes (n, distinct gold tools)",cnt)
    json.dump(dict(classes=cnt),open(f"/vol/fb3/{mname}_nat_classes{ptag}.json","w")); VC()
    thash=lambda n:h8("tool:"+n)%10
    def split_of(name): h=thash(name); return "ann" if h<4 else ("dev" if h<6 else "teach")
    def randtool(a,rng):
        names=[n for n in ALLTOOLS if n!=a["gold"][0]["name"] and n not in {t["name"] for t in a["item"]["tools"]}]
        return dict(a,gold=[dict(name=rng.choice(names),arguments=a["gold"][0]["arguments"])])
    def with_gold(a,gold): return dict(a,gold=gold,target=gtxt(gold)+"<|im_end|>")
    def make_ctl(anns,seed,kind):
        r=random.Random(seed*131+7); out=[]
        for i,a in enumerate(anns):
            if kind=="randtool": b=randtool(a,r); out.append(with_gold(b,b["gold"]))
            else:  # mixed: ~50% keep the right label, ~50% carry the base's own wrong call
                if r.random()<0.5 and a["wrong"]: out.append(with_gold(a,a["wrong"]))
                else: out.append(a)
        return out
    def apply(cfg,anns):
        S=sites(anns)
        if not S: return 0,[]
        m=cfg["m"]; ed=[]
        if m=="memit_down": memit(S,LS[cfg["ls"]],"down",cfg["lam"])
        elif m=="lmhead_lda": ed=lmhead(S,cfg["alpha"],"lda",cfg.get("ridge",0.1))
        elif m=="lmhead_plain": ed=lmhead(S,cfg["alpha"],"plain")
        elif m=="lmhead_cal": ed=lmhead_cal(S,cfg.get("ridge",0.1),cfg.get("margin",1.0))
        elif m=="lmhead_cal2": ed=lmhead_cal(S,cfg.get("ridge",0.1),cfg.get("margin",1.0),pos=True)
        return len(S),ed
    LS={"d3_80":[int(NL*.8)+i for i in range(3)]}
    def run_scn(scn_name,ann_pool,dev_pool,teach_pool,neigh_cands,ND,tagx):
        random.Random(3).shuffle(neigh_cands)
        teach=teach_pool[:100]; dev=dev_pool[:50]
        bt=ev(teach); bd=ev(dev)
        # neighbours: base-verified under THIS engine/batching
        tools_used={it["tool"] for it in ann_pool}|{it["tool"] for it in teach}
        pri=[n for n in neigh_cands if n["tool"] in tools_used]; rest=[n for n in neigh_cands if n["tool"] not in tools_used]
        cand=(pri+rest)[:700]; nv=ev(cand); neigh=[n for n,c in zip(cand,nv) if c][:400]; ndev=neigh[:80]
        log(scn_name,"ann_pool",len(ann_pool),"teach",len(teach),"dev",len(dev),"neigh verified",len(neigh),"of",len(cand),"(same-tool",sum(n['tool'] in tools_used for n in neigh),")","base teach",bt.mean() if len(bt) else None)
        R=dict(model=model,scen=scn_name,ntools_ann=len({a['tool'] for a in ann_pool}),sizes=dict(ann_pool=len(ann_pool),teach=len(teach),dev=len(dev),neigh=len(neigh)),base_teach=bt.tolist(),draws=[],sweep=[],codec=[],diag=[])
        def draw(seed):
            r=random.Random(seed); L=list(ann_pool); r.shuffle(L); return L[:25]
        cfgs=[dict(m="lmhead_lda",alpha=a,ridge=0.1) for a in [0.1,0.2,0.3,0.45,0.7]]+[dict(m="lmhead_plain",alpha=0.03),dict(m="lmhead_plain",alpha=0.01),dict(m="lmhead_cal"),dict(m="lmhead_cal2"),dict(m="memit_down",ls="d3_80",lam=1000.0)]
        if only: cfgs=[c for c in cfgs if c["m"] in only.split(",")]
        for cfg in cfgs:
            restore(); n,_=apply(cfg,draw(0)); d_=ev(dev); nd_=ev(ndev); restore()
            rec=dict(cfg=cfg,nsites=n,dev_gain=float(d_.mean()-bd.mean()) if len(d_) else None,neigh_dev=float(nd_.mean()) if len(nd_) else None); R["sweep"].append(rec); log(rec)
            json.dump(R,open(f"/vol/fb3/{tagx}.json","w")); VC()
        lda=[r for r in R["sweep"] if r["cfg"]["m"]=="lmhead_lda"]; okl=[r for r in lda if r["neigh_dev"]>=0.98] or lda
        bestlda=max(okl,key=lambda r:r["dev_gain"])["cfg"] if okl else dict(m="lmhead_lda",alpha=0.2,ridge=0.1); R["chosen_lda"]=bestlda
        methods={"lmhead_lda":bestlda,"lmhead_plain":dict(m="lmhead_plain",alpha=0.03),"lmhead_plain_xs":dict(m="lmhead_plain",alpha=0.01),"lmhead_cal":dict(m="lmhead_cal"),"lmhead_cal2":dict(m="lmhead_cal2"),"memit_down":dict(m="memit_down",ls="d3_80",lam=1000.0)}
        if only: methods={k:v for k,v in methods.items() if k in only.split(",")}
        for name,cfg in methods.items():
            for d in range(ND):
                anns=draw(d); rec=dict(method=name,cfg=cfg,draw=d,ann_tools=len({a["tool"] for a in anns}))
                restore(); n,ed=apply(cfg,anns); t,o=ev(teach,True); nb=ev(neigh); rec["real"]=dict(teach=t.tolist(),neigh=nb.tolist(),nsites=n,rows=len(ed))
                if d==0: R["diag"].append(dict(method=name,outs=o[:6],golds=[x["gold"][0]["name"] for x in teach[:6]]))
                restore()
                for ck in ["mixed","randtool"]:
                    n2,_=apply(cfg,make_ctl(anns,d,ck)); rec[ck]=dict(teach=ev(teach).tolist()); restore()
                R["draws"].append(rec); log(scn_name,name,d,"real",round(float(t.mean()),3),"neigh",round(float(nb.mean()),3),"mixed",round(float(np.mean(rec['mixed']['teach'])),3),"rand",round(float(np.mean(rec['randtool']['teach'])),3))
                json.dump(R,open(f"/vol/fb3/{tagx}.json","w")); VC()
        # ---- real-codec stand-in on lm_head (lmhead_lda best cfg), 3 draws
        try:
            restore(); saved.clear(); W0=lm.lm_head.weight.data.clone(); sg=(torch.randint(0,2,(D_h,),device=dev,generator=torch.Generator(device=dev).manual_seed(0))*2-1).float(); Lv=8
            A,hist0,s0=enc_rows(W0,sg,Lv); b0=ent_bytes(hist0,s0.numel()); fp16=W0.numel()*2
            lm.lm_head.weight.data.copy_(A.to(W0.dtype)); Aq=lm.lm_head.weight.data.clone(); qt=ev(teach); qn=ev(neigh)
            R["codec_real"]=dict(Lv=Lv,fp16_bytes=fp16,artifact_bytes_before=b0,bpw=b0*8/W0.numel(),base_teach=qt.tolist(),base_neigh=qn.tolist(),runs=[])
            for d in range(3):
                for var in ["reencode_refit","reencode_orig_scale"]:
                    lm.lm_head.weight.data.copy_(Aq); S=sites(draw(d)); ed=lmhead(S,bestlda["alpha"],"lda",0.1); w=lm.lm_head.weight.data
                    rows=torch.tensor(sorted(set(ed)),device=dev); Wr=w[rows].float(); ng=D_h//128
                    _,hb,sb=enc_rows(Aq[rows].float(),sg,Lv,None)   # old rows' histogram (re-encoded from artifact values)
                    sold=s0.reshape(-1,ng,1)[rows].reshape(-1,1) if var=="reencode_orig_scale" else None
                    Q,ha,sa=enc_rows(Wr,sg,Lv,sold); w[rows]=Q.to(w.dtype)
                    hist1=hist0-hb+ha; b1=ent_bytes(hist1,s0.numel()); t_=ev(teach); n_=ev(neigh)
                    R["codec_real"]["runs"].append(dict(var=var,draw=d,rows=len(ed),bytes_after=b1,teach=t_.tolist(),neigh=n_.tolist())); log("codec_real",var,d,t_.mean(),n_.mean(),"bytes",b0,"->",b1)
                    json.dump(R,open(f"/vol/fb3/{tagx}.json","w")); VC()
            saved.clear(); lm.lm_head.weight.data.copy_(W0)
        except Exception as e: log("codec_real failed",repr(e)[:300]); restore()
        R["done"]=True; json.dump(R,open(f"/vol/fb3/{tagx}.json","w")); VC(); log("DONE",tagx)
    @torch.no_grad()
    def pos_stats():
        ps=t1_cov[:100]; outs=gen(ps,40); xs=[]
        for p,o in zip(ps,outs):
            n=len(tok(p,add_special_tokens=False).input_ids); ids=torch.tensor([tok(p,add_special_tokens=False).input_ids+o],device=dev)
            fh={}; h=lm.model.norm.register_forward_hook(lambda m,i_,o_: fh.__setitem__("h",o_)); lm.model(ids); h.remove(); xs.append(fh["h"][0,n-1:-1].float())
        X=torch.cat(xs); return X.mean(0),torch.cov(X.T)
    for spec in scen.split(","):
        EXTRA[0]=True; G["mu"],G["sig"]=gstats(True); G["mu_pos"],G["sig_pos"]=pos_stats()
        if spec.startswith("nat"):
            cls=spec.split(":")[1] if ":" in spec else None
            order=sorted(cnt,key=lambda c:-cnt[c][1]) if cls is None else [cls]
            for c in order[:2 if cls is None else 1]:
                if cnt.get(c,(0,0))[0]<60 or cnt[c][1]<30: log("skip class",c,cnt.get(c)); continue
                pools={"ann":[],"dev":[],"teach":[]}
                for i in FAIL[c]:
                    x=byid[i]; calls,_=parse_calls(BASE[i]["text"]); wrong=[dict(name=n,arguments=a) for n,a in calls]
                    pools[split_of(x["gold"][0]["name"])].append(mk(x,x["gold"],wrong=wrong))
                fid=set(FAIL[c]); neigh=[mk(byid[i],byid[i]["gold"]) for i,v in BASE.items() if v["correct"] and i not in fid]
                run_scn(f"nat_{c}",pools["ann"],pools["dev"],pools["teach"],neigh,ndraw,f"{mname}_nat_{c}{ptag}")
        elif spec.startswith("pair"):
            pools={"ann":[],"dev":[],"teach":[]}; plain=[]
            for x in singles:
                hh=h8(x["id"])%10
                if hh==9: continue
                g=x["gold"][0]; gt=[t for t in x["tools"] if t["name"]==g["name"]]
                if not gt: continue
                twin=dict(gt[0],name=g["name"]+"_v2"); tools=list(x["tools"]); tools.insert(tools.index(gt[0])+(h8(x["id"]+"p")%2),twin)
                pools[("ann","dev","teach")[0 if hh<4 else (1 if hh<6 else 2)]].append(mk(x,[dict(name=twin["name"],arguments=g["arguments"])],tools=tools,wrong=[g]))
            neigh=[mk(byid[i],byid[i]["gold"]) for i,v in BASE.items() if v["correct"] and h8(i)%10 in (9,0,1,2,3)]
            run_scn("pairS",pools["ann"],pools["dev"],pools["teach"],neigh,ndraw,f"{mname}_pairS{ptag}")
        elif spec.startswith("new"):
            AN="escalate_to_specialist"; pools={"ann":[],"dev":[],"teach":[]}
            for x in nocall:
                pools[("ann","dev","teach")[0 if h8(x["id"])%10<4 else (1 if h8(x["id"])%10<6 else 2)]].append(mk(x,[dict(name=AN,arguments={})],wrong=[dict(name="forward_to_billing",arguments={})]))
            neigh=[mk(x,x["gold"]) for i,v in BASE.items() if v["correct"] for x in [byid[i]]]
            run_scn("newtool",pools["ann"],pools["dev"],pools["teach"],neigh,min(ndraw,5),f"{mname}_newtool{ptag}")
    return "ok"

@app.local_entrypoint()
def main(model:str="Qwen/Qwen3-8B",scen:str="nat",ptag:str="",ndraw:int=10,only:str=""):
    print(run.remote(model,scen,ptag,ndraw,only))
