import modal, json
app = modal.App("tenpct-v2-feedback")
img = (modal.Image.debian_slim().pip_install("torch","transformers==4.46.3","datasets","numpy","accelerate")
       .add_local_file("lex.py","/root/lex.py"))
vol = modal.Volume.from_name("hf-cache", create_if_missing=True)

@app.function(image=img, gpu="A10G", timeout=7200, volumes={"/hf":vol})
def run(model="Qwen/Qwen2.5-0.5B-Instruct", tag="dev"):
    import os, sys, time, random, numpy as np, torch
    os.environ["HF_HOME"]="/hf"; sys.path.insert(0,"/root")
    from datasets import load_dataset
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from lex import Memory
    t0=time.time()
    tok=AutoTokenizer.from_pretrained(model); tok.padding_side="right"
    lm=AutoModelForCausalLM.from_pretrained(model,torch_dtype=torch.bfloat16).cuda().eval()
    pos=tok.encode(" positive")[0]; neg=tok.encode(" negative")[0]
    W=lm.lm_head.weight[[neg,pos]].float()   # [2,d]
    HEAD="Classify the sentiment of the review as positive or negative.\n\n"
    def prompt(t,shots=()):
        return HEAD+"".join(f"Review: {s}\nSentiment: {'positive' if y else 'negative'}\n\n" for s,y in shots)+f"Review: {t}\nSentiment:"
    @torch.no_grad()
    def embed(prompts,bs=32):
        order=np.argsort([len(p) for p in prompts]); H=[None]*len(prompts); D=np.zeros(len(prompts),np.float32)
        for i in range(0,len(prompts),bs):
            idx=order[i:i+bs]; enc=tok([prompts[j] for j in idx],return_tensors="pt",padding=True).to("cuda")
            h=lm.model(**enc).last_hidden_state; last=enc.attention_mask.sum(1)-1
            hl=h[torch.arange(len(idx)),last].float(); lg=hl@W.T
            for a,j in enumerate(idx): H[j]=hl[a].cpu().numpy(); D[j]=(lg[a,1]-lg[a,0]).item()
        return np.stack(H),D
    # ---------------- data
    ds=load_dataset("glue","sst2")["train"]
    rows=[(r["sentence"].strip(),r["label"]) for r in ds if len(r["sentence"].split())>=6]
    random.Random(0).shuffle(rows)
    dev=rows[:1000]; tuneold=rows[1000:1500]; errpool=rows[1500:4000]; anchsrc=rows[4000:4600]
    WORDS=["blenky","zorbly","flumpy","grindle","snorfy","quixel","wumbly","drazzle","fennick","glorpy","tranty","vossy",
           "klindy","moopish","jarnful","splonky","brevic","yazzled","omnoxious","pliffy","gwibbly","thrunk","zeblish","crondle"]
    rr=random.Random(1); wl=[(w,rr.randint(0,1)) for w in WORDS]
    TPL=["The movie was totally {w}.","I thought the service here was {w}.","Honestly this product feels so {w}.","What a {w} performance by the whole cast.",
         "The food at that place is {w}, I tell you.","My experience with the support team was {w}.","Everyone said the new update is {w}.",
         "That concert last night was absolutely {w}.","I would describe the hotel as {w}.","The story, the acting, all of it: {w}.",
         "Not gonna lie, the delivery was kind of {w}.","Her new album is {w}.","This app has become really {w} lately.","We found the whole trip {w}."]
    PRE=["","","Update: ","Review: ","Just got back. ","Final verdict: "]
    def gen(words,n,seed):
        r=random.Random(seed); out=[]; seen=set()
        for w,y in words:
            c=0
            while c<n:
                s=r.choice(PRE)+r.choice(TPL).format(w=w)+r.choice(["", " Thanks.", " Would say so again."," Not sure what else to add."])
                if s in seen: continue
                seen.add(s); out.append((s,y,w)); c+=1
        return out
    TUNEW=wl[:6]; TESTW=wl[6:12]
    def slang(words,seed):
        allx=gen(words,100,seed); pool=[x for i,x in enumerate(allx) if i%100<60]; test=[x for i,x in enumerate(allx) if i%100>=60]
        return pool,test
    sc={}
    sl_tune=slang(TUNEW,11); sl_test=slang(TESTW,12)
    def feats(items):
        t=[x[0] for x in items]; H,D=embed([prompt(s) for s in t]); return dict(t=t,y=np.array([x[1] for x in items]),H=H,D=D,w=[x[2] if len(x)>2 else None for x in items])
    F={}
    F["dev"]=feats(dev); F["tuneold"]=feats(tuneold); F["err"]=feats(errpool); F["anch"]=feats(anchsrc)
    for nm,(p,t) in (("tune",sl_tune),("test",sl_test)): F[nm+"_pool"]=feats(p); F[nm+"_test"]=feats(t)
    for k,v in F.items(): print(k,"base acc",float(((v["D"]>0)==v["y"]).mean()),len(v["y"]),flush=True)
    mu=F["tuneold"]["H"].mean(0)
    def norm(H):
        Z=H-mu; return Z/np.linalg.norm(Z,axis=-1,keepdims=True)
    def q8(Z):  # int8 symmetric quantisation of unit vectors, return dequantised
        return np.clip(np.round(Z*127/np.abs(Z).max(-1,keepdims=True)),-127,127)*np.abs(Z).max(-1,keepdims=True)/127
    for v in F.values(): v["Z"]=norm(v["H"])
    BIG=8.0
    toks={}
    def ids(t):
        if t not in toks: toks[t]=tok.encode(t)
        return toks[t]
    acc=lambda D,y:float(((D>0)==y).mean())
    # --------------- methods: fn(mem, q) -> new diff ; mem: dict(idx into pool set P, P)
    def mk_mem(P,idx): return dict(t=[P["t"][i] for i in idx],y=P["y"][idx],Z=q8(P["Z"][idx]),H=P["H"][idx],D=P["D"][idx])
    def m_base(mem,q,pr): return q["D"]
    def m_cache(mem,q,pr):
        if len(mem["y"])==0: return q["D"]
        S=q["Z"]@mem["Z"].T; j=S.argmax(1); ok=S.max(1)>=pr["tau"]
        return np.where(ok,(2*mem["y"][j]-1)*BIG,q["D"])
    def m_knn(mem,q,pr):   # similarity-gated vote; override
        S=q["Z"]@mem["Z"].T; sg=2*mem["y"]-1
        wgt=np.where(S>=pr["tau"],np.exp((S-1)/pr.get("T",0.1)),0.0); v=wgt@sg
        return np.where(wgt.sum(1)>0,np.sign(v)*BIG+0*v,q["D"])
    def m_knnsoft(mem,q,pr):   # additive correction, base kept when evidence weak
        S=q["Z"]@mem["Z"].T; sg=2*mem["y"]-1
        wgt=np.maximum(S-pr["tau"],0)/(1-pr["tau"]); return q["D"]+pr["alpha"]*(wgt@sg)
    def m_bias(mem,q,pr):
        grid=np.linspace(-6,6,49); A=mem["_anch"]
        def sc(b): return ((mem["D"]+b>0)==mem["y"]).sum()*3+((A["D"]+b>0)==(A["D"]>0)).sum()
        b=max(grid,key=lambda b:(sc(b),-abs(b)))
        mem["bytes_extra"]=2; return q["D"]+b
    def m_ridge(mem,q,pr):   # residual linear head on centered hidden; closed form, anchors keep old behaviour
        A=mem["_anch"]; Xc=norm(mem["H"]); Xa=norm(A["H"]); sg=2*mem["y"]-1
        r=np.maximum(pr["margin"]-sg*mem["D"],0)*sg   # needed shift
        wc=pr["wc"]; X=np.vstack([Xc*np.sqrt(wc),Xa]); R=np.concatenate([r*np.sqrt(wc),np.zeros(len(Xa))])
        w=np.linalg.solve(X.T@X+pr["lam"]*np.eye(X.shape[1]),X.T@R)
        s=np.abs(w).max()/127; wq=np.round(w/s)*s; mem["bytes_extra"]=len(w)+2
        return q["D"]+norm(q["H"])@wq
    LEXCACHE={}
    def m_lex(mem,q,pr):
        A=mem["_anch"]; L=Memory(margin=pr["margin"],max_df=pr["max_df"],epochs=30,seed=0)
        for t,y,d in zip(A["t"],A["y"],A["D"]): L.add_anchor(ids(t),y,[-d/2,d/2])
        for t,y,d in zip(mem["t"],mem["y"],mem["D"]): L.add(ids(t),y,[-d/2,d/2])
        L.fit_pending(); mem["bytes_extra"]=L.bytes()-sum(2*len(a[0])+1 for a in L.anc)+0  # weights only; anchors counted separately
        mem["bytes_anch_lex"]=sum(2*len(a[0])+1 for a in L.anc)
        z=L.delta([ids(t) for t in q["t"]]); return q["D"]+z
    def m_icl(mem,q,pr):
        S=q["Z"]@mem["Z"].T; mx=S.max(1); sel=np.where(mx>=pr["tau"])[0]; out=q["D"].copy()
        if len(sel)==0: return out
        m=min(pr["m"],len(mem["y"])); P=[]
        for i in sel:
            nn=np.argsort(-S[i])[:m][::-1]; P.append(prompt(q["t"][i],[(mem["t"][j],int(mem["y"][j])) for j in nn]))
        _,D=embed(P); out[sel]=D; return out
    def m_icl_knn(mem,q,pr):  # ICL on gated queries, then hard kNN override at higher tau
        d=m_icl(mem,q,pr); return m_knn(mem,dict(q,D=d),dict(tau=pr["tau2"]))
    METH=dict(base=(m_base,[{}]),cache=(m_cache,[dict(tau=.99)]),
      knn=(m_knn,[dict(tau=t) for t in (.5,.6,.7,.8,.9)]),
      knnsoft=(m_knnsoft,[dict(tau=t,alpha=a) for t in (.3,.5,.7) for a in (2,6)]),
      bias=(m_bias,[{}]),
      ridge=(m_ridge,[dict(margin=m,wc=wc,lam=l) for m in (2,) for wc in (3,) for l in (.1,1)]),
      lex=(m_lex,[dict(margin=m,max_df=d) for m in (1,2) for d in (5,30)]),
      icl=(m_icl,[dict(tau=t,m=4) for t in (.3,.5,.7)]),
      icl_knn=(m_icl_knn,[dict(tau=t,m=4,tau2=t2) for t in (.5,) for t2 in (.8,.9)]))
    NANCH=200
    def run_one(name,pr,pool,heldout,k,seed,extra_test=None):
        fn=METH[name][0]; r=np.random.RandomState(seed)
        wrong=np.where((pool["D"]>0)!=pool["y"])[0]
        if pool["w"][0] is not None:   # round-robin over words
            by={}
            for i in r.permutation(wrong): by.setdefault(pool["w"][i],[]).append(i)
            ws=list(by); r.shuffle(ws); idx=[]; c=0
            while len(idx)<k and any(by.values()):
                w=ws[c%len(ws)]; c+=1
                if by[w]: idx.append(by[w].pop())
        else: idx=list(r.permutation(wrong)[:k])
        idx=np.array(idx,int); mem=mk_mem(pool,idx)
        ai=np.random.RandomState(100+seed).permutation(len(F["anch"]["y"]))[:NANCH]
        A=F["anch"]; mem["_anch"]=dict(t=[A["t"][i] for i in ai],y=A["y"][ai],H=A["H"][ai],D=A["D"][ai],Z=A["Z"][ai])
        mem["bytes_extra"]=0; mem["bytes_anch_lex"]=0
        res={}
        for nm,q in (("dev",F["dev"]),("held",heldout),("rec",dict(t=mem["t"],y=mem["y"],Z=pool["Z"][idx],H=pool["H"][idx],D=mem["D"]))):
            res[nm]=acc(fn(mem,q,pr),q["y"])
        # bytes
        d=F["dev"]["H"].shape[1]
        if name in("cache","knn","knnsoft"): b=len(idx)*(d+3)
        elif name=="icl" or name=="icl_knn": b=len(idx)*(d+3)+sum(len(t.encode()) for t in mem["t"])
        elif name=="base": b=0
        elif name=="bias": b=2
        elif name=="ridge": b=mem["bytes_extra"]+NANCH*(d+3)
        elif name=="lex": b=mem["bytes_extra"]+mem["bytes_anch_lex"]+len(idx)*0
        res["bytes"]=int(b)+(2*d if name not in("base","bias","lex") else 0)
        return res
    def sweep(name,prs,pool,held,ks,seeds):
        out=[]
        for pr in prs:
            for k in ks:
                rs=[run_one(name,pr,pool,held,k,s) for s in seeds]
                out.append(dict(m=name,pr=pr,k=k,**{key:float(np.mean([x[key] for x in rs])) for key in rs[0]}))
        return out
    base_dev=acc(F["dev"]["D"],F["dev"]["y"]); print("base dev",base_dev)
    # ---- hyperparameter tuning on TUNE words + tuneold (never test words)
    F["dev_backup"]=F["dev"]; F["dev"]=F["tuneold"]; bt=acc(F["tuneold"]["D"],F["tuneold"]["y"])
    best={}; tuning=[]
    for name,(fn,prs) in METH.items():
        if len(prs)==1: best[name]=prs[0]; continue
        sc=[]
        for pr in prs:
            rs=[run_one(name,pr,F["tune_pool"],F["tune_test"],k,s) for k in (5,10,25) for s in (0,1)]
            nw=np.mean([x["held"] for x in rs]); dv=np.mean([x["dev"] for x in rs])
            sc.append((nw-5*max(0,bt-dv-0.003),pr,nw,dv)); tuning.append(dict(m=name,pr=pr,held=nw,dev=dv))
            print("tune",name,pr,round(nw,3),round(dv,4),flush=True)
        best[name]=max(sc,key=lambda x:x[0])[1]
    print("BEST",best,flush=True)
    F["dev"]=F["dev_backup"]
    # ---- final eval
    ks=(1,5,10,25); seeds=list(range(5)); R=[]
    errw=np.where((F["err"]["D"]>0)!=F["err"]["y"])[0]; print("n base errors in errpool",len(errw))
    # real-error scenario: pool=errpool, held = other errors (excluded per-seed approx: use fixed half split)
    half=len(errw)//2; Ep={kk:(v[errw[:half]] if hasattr(v,'__len__') and not isinstance(v,list) else None) for kk,v in F["err"].items()}
    def sub(P,ix): return {kk:(v[ix] if isinstance(v,np.ndarray) else [v[i] for i in ix]) for kk,v in P.items()}
    Epool=sub(F["err"],errw[:half]); Ehold=sub(F["err"],errw[half:])
    Epool["w"]=[None]*len(Epool["t"])
    for name in METH:
        for k in ks:
            for scen,pool,held in (("slang",F["test_pool"],F["test_test"]),("realerr",Epool,Ehold)):
                rs=[run_one(name,best[name],pool,held,k,s) for s in seeds]
                R.append(dict(scen=scen,m=name,pr=best[name],k=k,**{key:float(np.mean([x[key] for x in rs])) for key in rs[0]},
                              dev_sd=float(np.std([x["dev"] for x in rs]))))
                print("RES",R[-1],flush=True)
    out=dict(model=model,base=dict(dev=base_dev,slang_test=acc(F["test_test"]["D"],F["test_test"]["y"]),errpool=acc(F["err"]["D"],F["err"]["y"]),n_err=int(len(errw))),best=best,tuning=tuning,results=R,secs=time.time()-t0)
    os.makedirs("/hf/fb",exist_ok=True); json.dump(out,open(f"/hf/fb/{tag}.json","w")); vol.commit()
    print("RESULT_JSON",json.dumps(out,default=float))

@app.local_entrypoint()
def main(model:str="Qwen/Qwen2.5-0.5B-Instruct",tag:str="dev"):
    run.spawn(model,tag)
