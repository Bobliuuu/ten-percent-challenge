import modal, json
app = modal.App("tenpct-v2-feedback-mmlu")
img = (modal.Image.debian_slim().pip_install("torch","transformers==4.46.3","numpy","accelerate")
       .add_local_file("/Users/jhome/ten-percent/v2/base/pool.jsonl","/root/pool.jsonl")
       .add_local_file("/Users/jhome/ten-percent/v2/base/dev.jsonl","/root/dev.jsonl")
       .add_local_file("/Users/jhome/ten-percent/v2/base/calib.jsonl","/root/calib.jsonl"))
vol = modal.Volume.from_name("hf-cache", create_if_missing=True)

@app.function(image=img, gpu="A10G", timeout=14400, volumes={"/hf":vol})
def run(model="Qwen/Qwen2.5-0.5B-Instruct", tag="dev05", npool=3500):
    import os, time, random, numpy as np, torch
    os.environ["HF_HOME"]="/hf"
    from transformers import AutoTokenizer, AutoModelForCausalLM
    t0=time.time()
    tok=AutoTokenizer.from_pretrained(model); tok.padding_side="left"
    lm=AutoModelForCausalLM.from_pretrained(model,torch_dtype=torch.bfloat16).cuda().eval()
    LID=[tok.encode(" "+l)[0] for l in "ABCD"]; W=lm.lm_head.weight[LID].float()
    NL=lm.config.num_hidden_layers; MID=NL//2
    ld=lambda f:[json.loads(l) for l in open(f)]
    dev=ld("/root/dev.jsonl"); pool=ld("/root/pool.jsonl"); calib=ld("/root/calib.jsonl")
    rs=random.Random(0); rs.shuffle(pool); rs.shuffle(calib)
    TP=pool[:1000]; FP=pool[1000:npool]; TO=calib[:500]; AN=calib[500:700]   # tune pool, final pool, tune-old, anchors
    def stem(r): return f"The following are multiple choice questions (with answers) about {r['subject'].replace('_',' ')}.\n\n"
    def body(r): return f"{r['question']}\n"+"".join(f"{l}. {c}\n" for l,c in zip("ABCD",r['choices']))+"Answer:"
    def prompt(r,shots=()):
        return stem(r)+"".join(body(s)+" "+"ABCD"[s['answer']]+"\n\n" for s in shots)+body(r)
    tok.truncation_side="left"
    @torch.no_grad()
    def embed(prompts,bs=16,feat=True):
        order=np.argsort([len(p) for p in prompts]); N=len(prompts); L=np.zeros((N,4),np.float32)
        HL=[None]*N; HM=[None]*N
        for i in range(0,N,bs):
            idx=order[i:i+bs]; enc=tok([prompts[j] for j in idx],return_tensors="pt",padding=True,truncation=True,max_length=1024).to("cuda")
            o=lm.model(**enc,output_hidden_states=feat); h=o.last_hidden_state[:,-1].float(); lg=(h@W.T).cpu().numpy()
            if feat:
                m=enc.attention_mask.unsqueeze(-1).float(); hm=o.hidden_states[MID].float(); hm=((hm*m).sum(1)/m.sum(1)).cpu().numpy(); h=h.cpu().numpy()
            for a,j in enumerate(idx):
                L[j]=lg[a]
                if feat: HL[j]=h[a]; HM[j]=hm[a]
        return (L,np.stack(HL),np.stack(HM)) if feat else L
    def shuf(r):   # rotate choices by 1 (same question, different letters)
        c=r['choices']; c2=c[1:]+c[:1]; return dict(r,choices=c2,answer=(r['answer']-1)%4)
    def feats(rows):
        L,HL,HM=embed([prompt(r) for r in rows]); return dict(r=rows,y=np.array([r['answer'] for r in rows]),L=L,HL=HL,HM=HM)
    F=dict(dev=feats(dev),tp=feats(TP),fp=feats(FP),to=feats(TO),an=feats(AN))
    for k,v in F.items(): print(k,"base acc",float((v['L'].argmax(1)==v['y']).mean()),len(v['y']),flush=True)
    # shuffled variants of pool (for rec_shuf)
    for k in ("tp","fp"):
        rows=[shuf(r) for r in F[k]['r']]; F[k]['shuf']=feats(rows)
    MU={}
    for kind in ("HL","HM"): MU[kind]=F['to'][kind].mean(0)
    def Zof(P,kind):
        Z=P[kind]-MU[kind]; return Z/np.linalg.norm(Z,axis=-1,keepdims=True)
    def q8(Z):
        s=np.abs(Z).max(-1,keepdims=True); return np.clip(np.round(Z*127/s),-127,127)*s/127
    for P in list(F.values())+[F['tp']['shuf'],F['fp']['shuf']]:
        for kind in ("HL","HM"): P["Z"+kind]=Zof(P,kind)
    D=F['dev']['HL'].shape[1]; BIG=20.0
    acc=lambda L,y:float((L.argmax(1)==y).mean())
    def mk_mem(P,idx):
        return dict(rows=[P['r'][i] for i in idx],y=P['y'][idx],L=P['L'][idx],ZHL=q8(P['ZHL'][idx]),ZHM=q8(P['ZHM'][idx]),HL=P['HL'][idx])
    def match(mem,q,S,tau,T):
        """vote matrix [nq,4]: memory items (sim>=tau) vote for the choice whose text equals their stored correct answer"""
        V=np.zeros((len(q['r']),4),np.float32)
        ans=[m['choices'][m['answer']] for m in mem['rows']]
        for i,r in enumerate(q['r']):
            pos={c:a for a,c in enumerate(r['choices'])}
            for j in np.where(S[i]>=tau)[0]:
                a=pos.get(ans[j])
                if a is not None: V[i,a]+=np.exp((S[i,j]-1)/T)
        return V
    def m_base(mem,q,pr): return q['L']
    def m_knn(mem,q,pr):
        kind=pr['kind']; S=q['Z'+kind]@mem['Z'+kind].T; V=match(mem,q,S,pr['tau'],0.1)
        return np.where(V.sum(1,keepdims=True)>0,V*BIG,q['L'])
    def m_knnsoft(mem,q,pr):
        kind=pr['kind']; S=q['Z'+kind]@mem['Z'+kind].T; V=match(mem,q,S,pr['tau'],0.1)
        return q['L']+pr['alpha']*V/ (np.exp(0)+0)
    def m_bias(mem,q,pr):   # 4-vector bias on letter logits, grid coordinate ascent; anchors keep old
        A=mem['_an']; b=np.zeros(4); y=mem['y']
        def sc(b): return 3*(( mem['L']+b).argmax(1)==y).sum()+((A['L']+b).argmax(1)==A['L'].argmax(1)).sum()
        cur=sc(b)
        for _ in range(3):
            for c in range(4):
                for v in np.linspace(-3,3,25):
                    bb=b.copy(); bb[c]=v; s=sc(bb)
                    if s>cur or (s==cur and abs(v)<abs(b[c])): b,cur=bb,s
        return q['L']+b
    def m_ridge(mem,q,pr):
        A=mem['_an']; Xc=Zof(dict(HL=mem['HL']),'HL'); Xa=A['ZHL']; y=mem['y']
        T=np.zeros((len(y),4)); 
        for i in range(len(y)): T[i,y[i]]=pr['margin']
        R=T-(mem['L']-mem['L'].mean(1,keepdims=True))*0   # target residual: boost true letter
        wc=pr['wc']; X=np.vstack([Xc*np.sqrt(wc),Xa]); Y=np.vstack([R*np.sqrt(wc),np.zeros((len(Xa),4))])
        w=np.linalg.solve(X.T@X+pr['lam']*np.eye(X.shape[1]),X.T@Y); s=np.abs(w).max()/127; w=np.round(w/s)*s
        return q['L']+q['ZHL']@w
    def m_icl(mem,q,pr):
        kind=pr['kind']; S=q['Z'+kind]@mem['Z'+kind].T; mx=S.max(1); sel=np.where(mx>=pr['tau'])[0]; out=q['L'].copy()
        if len(sel)==0: return out
        m=min(pr['m'],len(mem['y'])); P=[]
        for i in sel:
            nn=np.argsort(-S[i])[:m][::-1]; P.append(prompt(q['r'][i],[mem['rows'][j] for j in nn]))
        out[sel]=embed(P,bs=8,feat=False); return out
    METH=dict(base=(m_base,[{}]),
      knn=(m_knn,[dict(kind=kd,tau=t) for kd in ("HL","HM") for t in (.5,.7,.85,.95)]),
      knnsoft=(m_knnsoft,[dict(kind="HM",tau=t,alpha=a) for t in (.5,.7,.85) for a in (2,5)]),
      bias=(m_bias,[{}]),
      ridge=(m_ridge,[dict(margin=m,wc=3,lam=l) for m in (2,4) for l in (.1,1)]),
      icl=(m_icl,[dict(kind=kd,tau=t,m=4) for kd in ("HL","HM") for t in (.6,.8,.9)]))
    def itemsz(name,k,mem):
        txt=sum(len(json.dumps(dict(q=r['question'],c=r['choices'])).encode()) for r in mem['rows'])
        return dict(base=0,bias=16,
            knn=k*(D+3)+2*D+sum(len(r['choices'][r['answer']].encode()) for r in mem['rows']),
            knnsoft=k*(D+3)+2*D+sum(len(r['choices'][r['answer']].encode()) for r in mem['rows']),
            icl=k*(D+3)+2*D+txt, ridge=4*D+4+len(F['an']['y'])*(D+3)+2*D)[name]
    def itemsz14(name,k,mem,d=5120):
        txt=sum(len(json.dumps(dict(q=r['question'],c=r['choices'])).encode()) for r in mem['rows']); ans=sum(len(r['choices'][r['answer']].encode()) for r in mem['rows'])
        return dict(base=0,bias=16,knn=k*(d+3)+2*d+ans,knnsoft=k*(d+3)+2*d+ans,icl=k*(d+3)+2*d+txt,ridge=4*d+4+len(F['an']['y'])*(d+3)+2*d)[name]
    def run_one(name,pr,pool,k,seed,dev):
        fn=METH[name][0]; r=np.random.RandomState(seed)
        wrong=np.where(pool['L'].argmax(1)!=pool['y'])[0]; idx=r.permutation(wrong)[:k]
        mem=mk_mem(pool,idx); A=F['an']; mem['_an']=A
        subj={pool['r'][i]['subject'] for i in idx}
        rel=np.array([i for i,x in enumerate(pool['r']) if x['subject'] in subj and i not in set(idx)],int)
        if len(rel)>300: rel=r.permutation(rel)[:300]
        def sub(P,ix): return dict(r=[P['r'][i] for i in ix],y=P['y'][ix],L=P['L'][ix],ZHL=P['ZHL'][ix],ZHM=P['ZHM'][ix])
        res={}
        for nm,q in (("dev",dev),("rec",sub(pool,idx)),("rec_shuf",sub(pool['shuf'],idx)),("rel",sub(pool,rel))):
            res[nm]=acc(fn(mem,q,pr),q['y']); res[nm+"_base"]=acc(q['L'],q['y'])
        res['bytes']=itemsz(name,k,mem); res['bytes14b']=itemsz14(name,k,mem); return res
    mean=lambda rs,key:float(np.mean([x[key] for x in rs]))
    # ---- tune on tune pool + tune-old (calib500); never dev
    tdev=F['to']; best={}; tuning=[]
    for name,(fn,prs) in METH.items():
        if len(prs)==1: best[name]=prs[0]; continue
        sc=[]
        for pr in prs:
            rsl=[run_one(name,pr,F['tp'],k,s,tdev) for k in (5,25) for s in (0,1)]
            dd=mean(rsl,'dev_base')-mean(rsl,'dev')
            s_=(mean(rsl,'rec')+mean(rsl,'rec_shuf'))/2+2*(mean(rsl,'rel')-mean(rsl,'rel_base'))-5*max(0,dd-0.003)
            sc.append((s_,pr)); tuning.append(dict(m=name,pr=pr,score=s_,dev_drop=dd,rec=mean(rsl,'rec'),rec_shuf=mean(rsl,'rec_shuf'),rel_gain=mean(rsl,'rel')-mean(rsl,'rel_base')))
            print("tune",tuning[-1],flush=True)
        best[name]=max(sc,key=lambda x:x[0])[1]
    print("BEST",best,flush=True)
    R=[]
    for name in METH:
        for k in (1,5,10,25):
            rsl=[run_one(name,best[name],F['fp'],k,s,F['dev']) for s in range(5)]
            R.append(dict(m=name,pr=best[name],k=k,**{key:mean(rsl,key) for key in rsl[0]},dev_sd=float(np.std([x['dev'] for x in rsl]))))
            print("RES",R[-1],flush=True)
    out=dict(model=model,dim=D,base_dev=acc(F['dev']['L'],F['dev']['y']),best=best,tuning=tuning,results=R,secs=time.time()-t0)
    os.makedirs("/hf/fb",exist_ok=True); json.dump(out,open(f"/hf/fb/{tag}.json","w")); vol.commit()
    print("RESULT_JSON",json.dumps(out,default=float))

@app.local_entrypoint()
def main(model:str="Qwen/Qwen2.5-0.5B-Instruct",tag:str="dev05"):
    run.remote(model,tag)
