"""v2/entropy: dump GPTQ indices (14B) for sampled layers, then lossless-coding analysis.
Launch: modal run --detach run.py --mode selftest|dump|analyze
RESULT_JSON lines; results also in Volume 'hf-cache' under /hf/v2entropy/."""
import modal, json, sys
app = modal.App("tenpct-v2entropy")
vol = modal.Volume.from_name("hf-cache", create_if_missing=True)
base = (modal.Image.debian_slim().pip_install("torch","transformers==4.46.3","numpy","accelerate","numba")
       .add_local_file("coders.py","/root/coders.py").add_local_file("../engine/split.py","/root/split.py")
       .add_local_file("../base/calib.jsonl","/root/base/calib.jsonl"))
LAYERS=[0,1,8,9,24,25,40,47]
NR=2048
LIN=["self_attn.q_proj","self_attn.k_proj","self_attn.v_proj","self_attn.o_proj","mlp.gate_proj","mlp.up_proj","mlp.down_proj"]

@app.function(image=base, gpu="A100-80GB", timeout=14400, volumes={"/hf":vol})
def dump(cfg):
    import os, time, math, torch, numpy as np, torch.nn as nn
    os.environ["HF_HOME"]="/hf"
    from transformers import AutoTokenizer, AutoModelForCausalLM
    t0=time.time(); M="Qwen/Qwen2.5-14B-Instruct"; L=cfg["L"]; rot=cfg["rot"]; g=128; name=cfg["name"]
    tok=AutoTokenizer.from_pretrained(M); model=AutoModelForCausalLM.from_pretrained(M,torch_dtype=torch.bfloat16).cuda().eval(); model.config.use_cache=False
    calsrc=[json.loads(x) for x in open("/root/base/calib.jsonl")]
    def fmt(r):
        s=f"The following are multiple choice questions (with answers) about {r['subject'].replace('_',' ')}.\n\n{r['question']}\n"
        for l,c in zip("ABCD",r["choices"]): s+=f"{l}. {c}\n"
        return s+"Answer: "+"ABCD"[r["answer"]]
    rng=np.random.RandomState(0); txt=[fmt(r) for r in calsrc]; rng.shuffle(txt)
    stream=[]
    for s in txt: stream+=tok.encode(s+"\n\n")
    ncal=cfg.get("ncal",64); Ls=512; stream=torch.tensor(stream[:ncal*Ls]).view(-1,Ls)[:ncal]
    layers=model.model.layers
    ins=[];kw={}
    class C(nn.Module):
        def __init__(s,m): super().__init__(); s.m=m
        def forward(s,x,*a,**k): ins.append(x); kw.update(k); raise ValueError
    layers[0]=C(layers[0])
    for i in range(len(stream)):
        try: model(stream[i:i+1].cuda())
        except ValueError: pass
    layers[0]=layers[0].m
    PGRID=torch.tensor([1.0,0.9,0.8,0.7,0.62,0.55,0.5,0.45,0.4,0.36,0.32,0.28,0.25]).cuda()
    def qparams(x,Lv):
        half=(Lv-1)/2; am=x.abs().max(1,keepdim=True)[0].clamp(min=1e-8); best=None
        for p in PGRID:
            s=(am*p/half).half().float().clamp(min=1e-8); q=(torch.clamp(torch.round(x/s+half),0,Lv-1)-half)*s
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
    def rot_cols(Mx,sg):
        R,C_=Mx.shape; return ((Mx*sg).view(R,C_//HB,HB)@Hb).reshape(R,C_)
    def gptq(W,H,Lv,blocksize=128,damp=0.01):
        W=W.float().clone(); R,C_=W.shape; H=H.clone()
        dead=torch.diag(H)==0; H[dead,dead]=1; W[:,dead]=0
        H=H+damp*torch.mean(torch.diag(H))*torch.eye(C_,device=H.device)
        Hinv=torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(H)),upper=True)
        QI=torch.zeros(R,C_,dtype=torch.uint8,device=W.device); S=[]; s=None
        for i1 in range(0,C_,blocksize):
            i2=min(i1+blocksize,C_); W1=W[:,i1:i2].clone(); Err1=torch.zeros_like(W1); Hi=Hinv[i1:i2,i1:i2]
            for i in range(i2-i1):
                if (i1+i)%g==0:
                    s=qparams(torch.cat([W1[:,i:],W[:,i2:]],1)[:,:g],Lv); S.append(s.half().flatten())
                w=W1[:,i]; d=Hi[i,i]
                q,qi=qidx(w.unsqueeze(1),s,Lv); q=q.flatten()
                QI[:,i1+i]=qi.flatten().to(torch.uint8)
                err=(w-q)/d
                W1[:,i:]-=err.unsqueeze(1)@Hi[i,i:].unsqueeze(0); Err1[:,i]=err
            W[:,i2:]-=Err1@Hinv[i1:i2,i2:]
        return QI,torch.stack(S,1)
    groups=[("self_attn.q_proj",["self_attn.q_proj","self_attn.k_proj","self_attn.v_proj"]),("self_attn.o_proj",["self_attn.o_proj"]),
            ("mlp.gate_proj",["mlp.gate_proj","mlp.up_proj"]),("mlp.down_proj",["mlp.down_proj"])]
    cur=[x.clone() for x in ins]; od=f"/hf/v2entropy/{name}"; os.makedirs(od,exist_ok=True)
    for oli,l in enumerate(layers):
        if oli>max(LAYERS): break
        want=oli in LAYERS
        if want:
            mods={n:l.get_submodule(n) for n in LIN}; Hs={}; nsm={}
            def mkh(key):
                def hook(m,i,o):
                    x=i[0].reshape(-1,i[0].shape[-1]).float(); nsm[key]=nsm.get(key,0)+x.shape[0]; Hs[key]=Hs.get(key,0)+2*x.T@x
                return hook
            hds=[mods[k].register_forward_hook(mkh(k)) for k,_ in groups]
            with torch.no_grad():
                for x in cur: l(x,**kw)
            for hh in hds: hh.remove()
            for key,members in groups:
                Hh=Hs[key]/nsm[key]; Cc=Hh.shape[0]
                sg=(torch.randint(0,2,(Cc,),generator=torch.Generator().manual_seed(1000*oli+len(key)),dtype=torch.float32)*2-1).cuda() if rot else torch.ones(Cc).cuda()
                if rot:
                    H1=rot_cols(Hh,sg); Hr=rot_cols(H1.T.contiguous(),sg).T.contiguous()
                else: Hr=Hh
                for n in members:
                    W=mods[n].weight.data.float(); Wr=rot_cols(W,sg) if rot else W
                    QI,S=gptq(Wr,Hr,L)
                    np.savez(f"{od}/l{oli}_{n}.npz",q=QI[:NR].cpu().numpy(),s=S[:NR].cpu().numpy(),shape=np.array(W.shape))
            del Hs
        with torch.no_grad():
            for j,x in enumerate(cur):
                o=l(x,**kw); cur[j]=o[0] if isinstance(o,tuple) else o
        print("layer",oli,round(time.time()-t0),flush=True)
    vol.commit(); print("DUMPED",name,flush=True); return name

def _analyze_one(args):
    import numpy as np, sys
    sys.path.insert(0,"/root"); import coders as cd
    path,prevpath,L,full=args
    z=np.load(path); Q=z["q"]; S=z["s"]; shp=z["shape"]
    pQ=np.load(prevpath)["q"] if prevpath else None
    R,C=Q.shape; N=R*C; sym=Q.reshape(-1).astype(np.int64)
    res=dict(path=path.split("/")[-1],R=R,C=C,N=N,full_shape=shp.tolist(),fixed=float(np.ceil(np.log2(L))),H0=cd.H(sym,L))
    ctxs=cd.make_ctx(Q,L,S,pQ)
    res["adapt"]={}
    for k,(c,n) in ctxs.items():
        res["adapt"][k]=cd.adapt_cost(sym,c.astype(np.int64),n,L,1.0,1e9)/N
    # best contexts mixing
    names=[k for k in ["colhist*row","colhist*pos","colhist*rowhist","colhist*row*col","row","col","prevlayer*colhist*row"] if k in ctxs]
    if full and len(names)>=3:
        cs=np.stack([ctxs[k][0].astype(np.int64) for k in names]); ns=np.array([ctxs[k][1] for k in names],dtype=np.int64)
        res["mix_names"]=names; res["mix"]=cd.mix_cost(sym,cs,ns,L,1.0,1e9,0.002,ctxs["pos"][0].astype(np.int64),8)/N
    # scales: fp16 lossless: raw 16, exponent-only entropy, ctx on left neighbor
    s16=S.view(np.uint16); ex=((s16>>10)&31).astype(np.int64); man=(s16&1023)
    res["scale_bits_raw"]=16.0
    res["scale_exp_H0"]=cd.H(ex.reshape(-1),32); 
    exl=np.zeros_like(ex); exl[:,1:]=ex[:,:-1]; exu=np.zeros_like(ex); exu[1:]=ex[:-1]
    ctx=((exl-exl.min()).clip(0,31)).reshape(-1).astype(np.int64)
    res["scale_exp_ctx_left"]=cd.adapt_cost(ex.reshape(-1),ctx,32,32,1.0,1e9)/ex.size
    res["scale_man_H0_top3"]=cd.H((man>>7).reshape(-1).astype(np.int64),8)
    res["scale_lossless_bits_per_scale"]=res["scale_exp_ctx_left"]+10.0
    # real coder verify on best single ctx
    best=min(res["adapt"],key=res["adapt"].get); c,n=ctxs[best]; c=c.astype(np.int64)
    if full:
        buf=cd.rc_encode(sym,c,n,L,24,65535); dec=cd.rc_decode(buf,N,c,n,L,24,65535)
        res["rc"]=dict(ctx=best,bytes=int(len(buf)),bits_per_param=len(buf)*8/N,decode_ok=bool((dec==sym).all()))
        ci=cd.adapt_cost(sym,c,n,L,24.0,65535.0)/N; res["rc"]["ideal_same_model_bpp"]=ci
    return res

@app.function(image=base, cpu=16, memory=65536, timeout=14400, volumes={"/hf":vol})
def analyze(name, L):
    import os, glob, json
    from multiprocessing import Pool
    vol.reload()
    od=f"/hf/v2entropy/{name}"; jobs=[]
    for f in sorted(glob.glob(od+"/*.npz")):
        b=os.path.basename(f); li=int(b.split("_")[0][1:]); rest=b.split("_",1)[1]
        pv=f"{od}/l{li-1}_{rest}"; pv=pv if os.path.exists(pv) else None
        jobs.append((f,pv,L,True))
    with Pool(16) as p: rs=p.map(_analyze_one,jobs,chunksize=1)
    json.dump(rs,open(f"{od}/analysis.json","w")); vol.commit()
    tot=sum(r["N"] for r in rs)
    summ={"name":name,"L":L,"n_params_sampled":tot}
    w=lambda f: sum(r["N"]*f(r) for r in rs)/tot
    summ["fixed"]=w(lambda r:r["fixed"]); summ["H0_per_module"]=w(lambda r:r["H0"])
    for k in rs[0]["adapt"]:
        if all(k in r["adapt"] for r in rs): summ["adapt_"+k]=w(lambda r:r["adapt"][k])
    summ["mix"]=w(lambda r:r.get("mix",float("nan")))
    summ["scale_exp_H0"]=w(lambda r:r["scale_exp_H0"]); summ["scale_exp_ctx_left"]=w(lambda r:r["scale_exp_ctx_left"])
    summ["rc_all_ok"]=all(r["rc"]["decode_ok"] for r in rs); summ["rc_bpp"]=w(lambda r:r["rc"]["bits_per_param"])
    summ["rc_ideal_bpp"]=w(lambda r:r["rc"]["ideal_same_model_bpp"])
    print("SUMMARY_JSON "+json.dumps(summ),flush=True)
    print("RESULT_JSON "+json.dumps(rs),flush=True)
    return summ

@app.function(image=base, timeout=3600)
def run_cfg(cfg):
    dump.remote(cfg); return analyze.remote(cfg["name"],cfg["L"])

@app.function(image=base, cpu=4, timeout=3600)
def selftest():
    import sys, numpy as np; sys.path.insert(0,"/root"); import coders as cd
    rng=np.random.RandomState(0); L=8; R,C=256,512
    Q=np.clip(np.round(rng.randn(R,C)*1.2+3.5),0,7).astype(np.uint8)
    S=(rng.rand(R,C//128)*0.1+0.01).astype(np.float16)
    ctxs=cd.make_ctx(Q,L,S,Q); sym=Q.reshape(-1).astype(np.int64)
    for k,(c,n) in ctxs.items(): print(k,cd.adapt_cost(sym,c.astype(np.int64),n,L,1.0,1e9)/sym.size)
    c,n=ctxs["colhist*row"]; c=c.astype(np.int64)
    buf=cd.rc_encode(sym,c,n,L,24,65535); dec=cd.rc_decode(buf,sym.size,c,n,L,24,65535)
    print("rc bpp",len(buf)*8/sym.size,"ok",bool((dec==sym).all()),"H0",cd.H(sym,L))
    r=_analyze_one.__wrapped__ if hasattr(_analyze_one,"__wrapped__") else None
    return "ok"

@app.function(image=base, timeout=14400)
def orch(cfgs):
    hs=[run_cfg.spawn(c) for c in cfgs]; out=[]
    for h in hs:
        try: out.append(h.get())
        except Exception as e: out.append({"error":repr(e)})
    print("ORCH_DONE "+json.dumps(out),flush=True); return out

@app.local_entrypoint()
def main(mode:str="selftest", cfgs:str=""):
    if mode=="selftest": print(selftest.remote()); return
    cs=json.load(open(cfgs)); orch.remote(cs)
