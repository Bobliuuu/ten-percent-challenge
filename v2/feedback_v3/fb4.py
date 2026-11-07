"""feedback_v3 round 2 (fb4): task B rerun with proper swap control, T1 real tool-use targets (S1-dev/S3-dev only),
lm_head codec survival, widened lmhead_lda alpha/ridge sweep, lmhead_lda + late-layer down_proj blend.
modal run --detach fb4.py --model Qwen/Qwen3-8B --scen T1pairS:first,T1pairP:first,T1new:first,T1pairS:last,T1new:last
scenarios: A, B (proxy data cached by fb3 on the volume), T1pairS (alias name = G+'_v2', distinguishing token LATER), T1pairP (alias = 'v2_'+G, FIRST token),
T1new (nocall queries -> call to NEW tool escalate_to_specialist not in the tool list). fmt first = native {"name":..,"arguments":..}; last = {"arguments":..,"name":..}.
Fitted-to-data quantities in the lmhead_lda edit: see README (generic covariance Sigma and mean mu from unlabeled text; alpha, ridge chosen on dev; class means from the <=25 annotations)."""
import modal, os, json
HERE=os.path.dirname(os.path.abspath(__file__)); T1=os.path.expanduser("~/ten-percent/v2/tournament/T1")
app=modal.App("tenpct-fb4")
img=(modal.Image.debian_slim(python_version="3.11").pip_install("torch==2.5.1","transformers==4.56.2","accelerate","numpy","datasets")
     .add_local_file(f"{T1}/scorer.py","/app/scorer.py").add_local_file(f"{T1}/data/s1_dev.jsonl","/app/s1_dev.jsonl")
     .add_local_file(f"{T1}/data/s3_dev.jsonl","/app/s3_dev.jsonl").add_local_file(f"{T1}/data/s1_calib.jsonl","/app/s1_calib.jsonl"))
vol=modal.Volume.from_name("tenpct-v2-hf")
GPU=os.environ.get("FB4_GPU","A100-80GB")

@app.function(image=img,gpu=GPU,timeout=36000,volumes={"/vol":vol})
def run(model="Qwen/Qwen3-8B",scen="T1pairS:first",ptag=""):
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
    # ---------------- scenarios
    SCN={}
    def build(scn,fmt):
        name=scn; EXTRA[0]=False
        items={"ann":[],"dev":[],"teach":[],"neigh_pool":[]}
        if scn in ("A","B"):
            tk=("A" if scn=="A" else "B"); D=json.load(open(f"/vol/fb3/data_{mname}_{tk}.json"))
            if scn=="B":
                TOOLS="""get_order_status(order_id): Look up the current status of an order.
cancel_order(order_id): Cancel an order that has not shipped yet.
refund_payment(order_id, amount): Return money to the customer's original payment method.
issue_store_credit(customer_id, amount): Add credit to the customer's store account balance.
update_shipping_address(order_id, address): Change the delivery address of an order.
cancel_subscription(subscription_id): Stop a recurring subscription.
close_account(customer_id): Permanently close a customer account.
escalate_to_human(reason): Hand the conversation to a human agent."""
                P=lambda t: chat("You are a support bot with these tools:\n"+TOOLS+"\n\nRespond with only a JSON tool call of the form {\"name\": <tool>, \"arguments\": {...}}.\n\nCustomer message: "+t)
                spl=lambda s:("ann","dev","teach")[0 if h8(s)%10<4 else (1 if h8(s)%10<6 else 2)]
                mk=lambda t,gold: dict(prompt=P(t),gold=gold,target='{"name": "'+gold+'"',cat=gold,txt=t)
                for t in D["pat"]: items[spl(t)].append(mk(t,"issue_store_credit"))
                for k,L in D["intent"].items():
                    for t in L: items["neigh_pool"].append(mk(t,k))
                def sc(it,o): m=re.search(r'"name"\s*:\s*"([a-z_]+)"',dec(o)); return (m.group(1) if m else None)==it["gold"]
                def shuf(anns,seed):
                    return [dict(a,gold="refund_payment",target='{"name": "refund_payment"') for a in anns]
                mx=48
            else:
                qs=D["old_names"]; P=lambda t: chat("You are routing customer support tickets. Pick the one best queue for the ticket from: "+", ".join(qs)+".\nReply with the queue name only.\n\nTicket: "+t)
                spl=lambda s:("ann","dev","teach")[0 if h8(s)%10<4 else (1 if h8(s)%10<6 else 2)]
                mk=lambda t,gold: dict(prompt=P(t),gold=gold,target=gold+"<|im_end|>",cat=gold,txt=t)
                for lab,L in D["new"].items():
                    for t in L: items[spl(t)].append(mk(t,lab))
                for lab,L in D["old"].items():
                    for t in L: items["neigh_pool"].append(mk(t,lab))
                def sc(it,o): return dec(o).lower().strip(" .")==it["gold"].lower()
                def shuf(anns,seed):
                    r=random.Random(seed+99); g=[a["gold"] for a in anns]
                    for _ in range(50):
                        r.shuffle(g)
                        if all(x!=a["gold"] for x,a in zip(g,anns)): break
                    return [dict(a,gold=x,target=x+"<|im_end|>") for a,x in zip(anns,g)]
                mx=10
            cap=(10,5) if scn=="A" else (100,60)
        else:
            base,kind=scn.rstrip(),None
            raw=[json.loads(l) for f in ("/app/s1_dev.jsonl","/app/s3_dev.jsonl") for l in open(f)]
            singles=[x for x in raw if x["kind"]=="single" and len(x["gold"])==1 and len(x["tools"])<=6]
            nocall=[x for x in raw if x["kind"]=="nocall" and len(x["tools"])<=6]
            tl=lambda x:[{"type":"function","function":t} for t in x["tools"]]
            gtxt=lambda calls: "\n".join("<tool_call>\n"+json.dumps({"arguments":c["arguments"],"name":c["name"]} if fmt=="last" else {"name":c["name"],"arguments":c["arguments"]},ensure_ascii=False)+"\n</tool_call>" for c in calls)
            def mk(x,gold,tools):
                it=dict(x,gold=gold,tools=tools); return dict(prompt=chat(x["query"],[{"type":"function","function":t} for t in tools],fmt),gold=gold,item=it,target=gtxt(gold)+"<|im_end|>",cat=scn,txt=x["query"])
            AN="escalate_to_specialist"
            for x in singles:
                hh=h8(x["id"])%10
                if hh==9: items["neigh_pool"].append(mk(x,x["gold"],x["tools"])); continue
                if scn.startswith("T1pair"):
                    g=x["gold"][0]; al=(g["name"]+"_v2") if scn=="T1pairS" else ("v2_"+g["name"])
                    gt=[t for t in x["tools"] if t["name"]==g["name"]]
                    if not gt: continue
                    twin=dict(gt[0],name=al); tools=list(x["tools"]); pos=tools.index(gt[0])+(h8(x["id"]+"p")%2)
                    tools.insert(pos,twin); items[("ann","dev","teach")[0 if hh<4 else (1 if hh<6 else 2)]].append(mk(x,[dict(name=al,arguments=g["arguments"])],tools))
            if scn=="T1new":
                for x in nocall:
                    hh=h8(x["id"])%10
                    if hh==9: continue
                    items[("ann","dev","teach")[0 if hh<4 else (1 if hh<6 else 2)]].append(mk(x,[dict(name=AN,arguments={})],x["tools"]))
            def sc(it,o): return bool(score_item(dec(o),it["item"])["correct"])
            def shuf(anns,seed):
                out=[]
                for a in anns:
                    if scn=="T1new": gold=[dict(name="forward_to_billing",arguments={})]
                    else: gold=[dict(name=a["gold"][0]["name"].replace("_v2","") if scn=="T1pairS" else a["gold"][0]["name"][3:],arguments=a["gold"][0]["arguments"])]
                    out.append(dict(a,gold=gold,target=gtxt(gold)+"<|im_end|>"))
                return out
            mx=100; cap=(60,40)
            EXTRA[0]=True
        rnd=random.Random(1)
        for k in items: rnd.shuffle(items[k])
        if cap and scn=="A":
            def bal(L,per):
                c={};o=[]
                for it in L:
                    if c.get(it["cat"],0)<per: c[it["cat"]]=c.get(it["cat"],0)+1;o.append(it)
                return o
            items["teach"]=bal(items["teach"],cap[0]); items["dev"]=bal(items["dev"],cap[1])
        elif cap: items["teach"]=items["teach"][:cap[0]]; items["dev"]=items["dev"][:cap[1]]
        items["ann"]=items["ann"][:400]; items["neigh_pool"]=items["neigh_pool"][:420]
        return items,sc,shuf,mx
    def pick_ann(wrong,seed,scn):
        r=random.Random(seed)
        if scn=="A":
            by={}
            for it in wrong: by.setdefault(it["cat"],[]).append(it)
            out=[]
            for c,L in by.items(): r.shuffle(L); out+=L[:2]
            rest=[it for it in wrong if it not in out]; r.shuffle(rest); out+=rest[:25-len(out)]; return out[:25]
        L=list(wrong); r.shuffle(L); return L[:25]
    # ---------------- run each scenario
    for spec in scen.split(","):
        scn,fmt=(spec.split(":")+["first"])[:2]; tag=f"{mname}_{scn}_{fmt}{ptag}"
        items,sc,shuf,MX=build(scn,fmt); BS["v"]=16 if scn.startswith("T1") else 48
        G["mu"],G["sig"]=gstats(EXTRA[0])
        def ev(L): 
            if not L: return np.zeros(0,int)
            o=gen([it["prompt"] for it in L],MX); return np.array([int(sc(it,x)) for it,x in zip(L,o)])
        nb=ev(items["neigh_pool"]); npool=[it for it,c in zip(items["neigh_pool"],nb) if c]; random.Random(2).shuffle(npool)
        items["neigh_dev"]=npool[:80]; items["neigh"]=npool[80:180]
        ap=ev(items["ann"]); wrong=[it for it,c in zip(items["ann"],ap) if not c]
        base_teach=ev(items["teach"]); base_dev=ev(items["dev"])
        log(spec,"sizes",{k:len(v) for k,v in items.items()},"ann_wrong",len(wrong),len(items["ann"]),"base teach",base_teach.mean() if len(base_teach) else None,"dev",base_dev.mean() if len(base_dev) else None,"neigh base-right",len(npool),"/",len(items["neigh_pool"]))
        R=dict(model=model,scen=scn,fmt=fmt,sizes={k:len(v) for k,v in items.items()},base_teach=base_teach.tolist(),base_dev=base_dev.tolist(),sweep=[],final=[],codec=[])
        LS={"d3_80":[int(NL*.8)+i for i in range(3)],"d3_65":[int(NL*.65)+i for i in range(3)]}
        cfgs=[]
        for a in [0.1,0.2,0.3,0.45,0.7,1.2,2.0]:
            for rg in [1e-2,1e-1]: cfgs.append(dict(m="lmhead_lda",alpha=a,ridge=rg))
        cfgs+=[dict(m="lmhead_mean",alpha=0.2),dict(m="memit_down",ls="d3_80",lam=1000.0)]
        for a in [0.2,0.45,1.0]: cfgs.append(dict(m="blend",ls="d3_80",lam=1000.0,alpha=a,ridge=1e-1))
        if os.environ.get("FB4_ONLY")=="plain": cfgs=[dict(m="lmhead_plain",alpha=a) for a in (0.1,0.2,0.4)]
        def apply(cfg,anns):
            S=sites(anns)
            if not S: return 0,[]
            m=cfg["m"]; ed=[]
            if m=="memit_down": memit(S,LS[cfg["ls"]],"down",cfg["lam"])
            elif m in ("lmhead_lda","lmhead_mean","lmhead_plain"): ed=lmhead(S,cfg["alpha"],{"lmhead_lda":"lda","lmhead_mean":"mean","lmhead_plain":"plain"}[m],cfg.get("ridge",1e-2))
            elif m=="blend":
                memit(S,LS[cfg["ls"]],"down",cfg["lam"],margin=1.0); S2=sites(anns); ed=lmhead(S2,cfg["alpha"],"lda",cfg["ridge"]) if S2 else []
            return len(S),ed
        def run_cfg(cfg,anns,keys):
            restore(); n,_=apply(cfg,anns); r={k:ev(items[k]) for k in keys}; restore(); return r,n
        ann0=pick_ann(wrong,0,scn)
        for cfg in cfgs:
            try:
                r,n=run_cfg(cfg,ann0,["dev","neigh_dev"]); rec=dict(cfg=cfg,nsites=n,dev=float(r["dev"].mean()),gain=float(r["dev"].mean()-base_dev.mean()),neigh_dev=float(r["neigh_dev"].mean()))
            except Exception as e: rec=dict(cfg=cfg,err=repr(e)[:300])
            R["sweep"].append(rec); log(rec); json.dump(R,open(f"/vol/fb3/{tag}.json","w")); VC()
        def best_of(pred):
            c=[r for r in R["sweep"] if "err" not in r and pred(r["cfg"]["m"])]
            if not c: return None
            ok=[r for r in c if r["neigh_dev"]>=0.98]
            return max(ok or c,key=lambda r:r["gain"]-(0 if ok else 10*(0.98-r["neigh_dev"])))["cfg"]
        chosen={"lmhead_lda":best_of(lambda m:m=="lmhead_lda"),"blend":best_of(lambda m:m=="blend"),"memit_down":best_of(lambda m:m=="memit_down")}
        if os.environ.get("FB4_ONLY")=="plain": chosen={"lmhead_plain":dict(m="lmhead_plain",alpha=0.2)}
        R["chosen"]=chosen; log("chosen",chosen)
        for name,cfg in chosen.items():
            if cfg is None: continue
            for seed in [0,1,2]:
                anns=pick_ann(wrong,seed,scn); rec=dict(method=name,cfg=cfg,seed=seed)
                for ctl,A in [("real",anns),("shuffled",shuf(anns,seed))]:
                    restore(); n,_=apply(cfg,A); t=ev(items["teach"]); nbv=ev(items["neigh"]); dr=drift() if (ctl=="real" and scn in("A","B","T1pairS")) else None; restore()
                    rec[ctl]=dict(teach=t.tolist(),neigh=nbv.tolist(),nsites=n,drift=dr)
                R["final"].append(rec); log(name,seed,"real",np.mean(rec["real"]["teach"]),"neigh",np.mean(rec["real"]["neigh"]),"shuf",np.mean(rec["shuffled"]["teach"]))
                json.dump(R,open(f"/vol/fb3/{tag}.json","w")); VC()
        # ---- codec test: lm_head quantised 4-bit g128 (stage-1 artifact), edit lmhead_lda, variants for the edited rows
        cfg=chosen.get("lmhead_lda")
        if cfg and (scn in("B","A","T1pairS","T1new") ):
            restore(); W0=lm.lm_head.weight.data.clone(); sc0,mn0=q4rows(W0); lm.lm_head.weight.data.copy_(q4apply(W0,sc0,mn0).to(W0.dtype))
            Wq=lm.lm_head.weight.data.clone()
            qt=ev(items["teach"]); qn=ev(items["neigh"]); R["codec_base"]=dict(teach=qt.tolist(),neigh=qn.tolist())
            for var in ["fp16row","req4_newscale","snap_orig"]:
                for seed in [0,1,2]:
                    anns=pick_ann(wrong,seed,scn); lm.lm_head.weight.data.copy_(Wq)
                    S=sites(anns); ed=lmhead(S,cfg["alpha"],"lda",cfg["ridge"]); w=lm.lm_head.weight.data
                    if var!="fp16row":
                        for t in ed:
                            row=w[t][None].float()
                            if var=="req4_newscale": s_,m_=q4rows(row)
                            else: s_,m_=sc0[t][None],mn0[t][None]
                            w[t]=q4apply(row,s_,m_)[0].to(w.dtype)
                    t_=ev(items["teach"]); n_=ev(items["neigh"])
                    badd=0 if var!="fp16row" else len(ed)*(D_h*2-D_h*0.5-(D_h/128)*4)
                    R["codec"].append(dict(var=var,seed=seed,teach=t_.tolist(),neigh=n_.tolist(),rows_edited=len(ed),bytes_added=badd)); log("codec",var,seed,t_.mean(),n_.mean(),len(ed),badd)
                    json.dump(R,open(f"/vol/fb3/{tag}.json","w")); VC()
            lm.lm_head.weight.data.copy_(W0)
        R["done"]=True; R["secs"]=time.time()-T0; json.dump(R,open(f"/vol/fb3/{tag}.json","w")); VC(); log("DONE",tag)
    return "ok"

@app.local_entrypoint()
def main(model:str="Qwen/Qwen3-8B",scen:str="T1pairS:first",ptag:str=""):
    print(run.remote(model,scen,ptag))
