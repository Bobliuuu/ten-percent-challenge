# Track Q harness: training-free compression sweep for vision classifiers. Run: modal run q.py --spec '<json>'
import modal, json, hashlib
app = modal.App("tenpct-trackQ")
img = (modal.Image.debian_slim().pip_install("torch","torchvision","transformers","datasets","timm","numpy","pillow","constriction")
       .add_local_file("/Users/jhome/.cache/huggingface/token","/root/hf_token"))

def hsplit(i): return int(hashlib.sha1(f"tenpct-{i}".encode()).hexdigest(),16)%2   # 0=dev (choices), 1=exam (locked)

@app.function(image=img, gpu=["A10G","L4","L40S","A100-40GB"], timeout=3000, max_containers=20, secrets=[])
def run(spec):
    import os, torch, numpy as np, torch.nn as nn, torch.nn.functional as F, time
    os.environ["HF_TOKEN"]=open("/root/hf_token").read().strip()
    from datasets import load_dataset
    torch.backends.cuda.matmul.allow_tf32=True; torch.backends.cudnn.allow_tf32=True
    dev="cuda"
    # ---- model
    mid=spec["model"]; kind=spec["kind"]
    if kind=="hf":
        from transformers import AutoModelForImageClassification, AutoImageProcessor
        m=AutoModelForImageClassification.from_pretrained(mid).float().eval().to(dev)
        pr=AutoImageProcessor.from_pretrained(mid); mean,std=pr.image_mean,pr.image_std
        sz=spec.get("size") or (pr.size.get("height") or pr.size.get("shortest_edge"))
        fwd=lambda x: m(x).logits
    else:
        import timm
        if mid.startswith("edadaltocg/"):
            from huggingface_hub import hf_hub_download
            cj=json.load(open(hf_hub_download(mid,"config.json"))); arch=cj["architecture"].replace("timm/","").split(".")[0]
            m=timm.create_model(arch,pretrained=False,num_classes=cj["num_classes"])
            if arch.startswith("resnet"):
                m.conv1=nn.Conv2d(3,64,3,1,1,bias=False); m.maxpool=nn.Identity()
            m.load_state_dict(torch.load(hf_hub_download(mid,"pytorch_model.bin"),map_location="cpu")); m=m.float().eval().to(dev)
            pc=cj["pretrained_cfg"]; mean,std=pc["mean"],pc["std"]; sz=spec.get("size") or pc["input_size"][-1]
        else:
            m=timm.create_model("hf-hub:"+mid,pretrained=True).float().eval().to(dev)
            c=timm.data.resolve_data_config({},model=m); mean,std=c["mean"],c["std"]; sz=spec.get("size") or c["input_size"][-1]
        fwd=lambda x: m(x)
    mean=torch.tensor(mean,device=dev).view(1,3,1,1); std=torch.tensor(std,device=dev).view(1,3,1,1)
    # ---- data
    ds=spec["data"]; ic,lc=ds["img"],ds["label"]
    from PIL import Image
    def tens(rows):
        out=[]
        for r in rows:
            r=r.convert("RGB")
            if ds.get("imagenet"):
                w,h=r.size; sc=256/min(w,h); r=r.resize((max(256,round(w*sc)),max(256,round(h*sc))),Image.BICUBIC); w,h=r.size
                l,t=(w-224)//2,(h-224)//2; r=r.crop((l,t,l+224,t+224))
            out.append(np.array(r))
        return torch.from_numpy(np.stack(out)).permute(0,3,1,2).contiguous()
    ncal=spec.get("ncal",256)
    if ds.get("imagenet"):
        te=load_dataset(ds["name"],data_files={"validation":"data/validation-*.parquet"},split="validation"); perm=np.random.RandomState(0).permutation(len(te)).tolist()
        pool=perm[:10000]; calix=perm[10000:10000+ncal]   # calibration disjoint from the 10k eval pool
        idx=[pool[i] for i in range(len(pool)) if hsplit(i)==0]; idx_ex=[pool[i] for i in range(len(pool)) if hsplit(i)==1]
        cal=te.select(calix)
    else:
        te=load_dataset(ds["name"],split=ds["test"]); idx=[i for i in range(len(te)) if hsplit(i)==0]; idx_ex=[i for i in range(len(te)) if hsplit(i)==1]
        cal=load_dataset(ds["name"],split=ds["train"]).shuffle(seed=0).select(range(ncal))
    def getsplit(ix):
        sub=te.select(ix); return tens(sub[ic]), torch.tensor(sub[lc])
    Xd,Yd=getsplit(idx); Xe,Ye=getsplit(idx_ex); Xc=tens(cal[ic])
    def prep(xb):
        x=xb.to(dev).float()/255; 
        if x.shape[-1]!=sz: x=F.interpolate(x,size=(sz,sz),mode="bicubic",align_corners=False).clamp(0,1)
        return (x-mean)/std
    @torch.no_grad()
    def predict(X):
        out=[]
        for i in range(0,len(X),200): out.append(fwd(prep(X[i:i+200])).argmax(-1).cpu())
        return torch.cat(out)
    base_sd={k:v.clone() for k,v in m.state_dict().items()}
    nparams=sum(p.numel() for p in m.parameters())
    base_pred={"dev":predict(Xd),"exam":predict(Xe)}
    Y={"dev":Yd,"exam":Ye}; X={"dev":Xd,"exam":Xe}
    res=dict(model=mid,params=nparams,fp32_bytes=nparams*4,base_acc={k:float((base_pred[k]==Y[k]).float().mean()) for k in Y},cfgs=[])
    targets=[(n,mod) for n,mod in m.named_modules() if (isinstance(mod,nn.Linear) or (isinstance(mod,nn.Conv2d) and mod.groups==1)) and not (n==spec.get("head","classifier"))]
    # head (last linear) kept fp16
    lastlin=[n for n,mod in m.named_modules() if isinstance(mod,nn.Linear)][-1]
    targets=[(n,mod) for n,mod in targets if n!=lastlin]
    # ---- Hessians / act norms
    H={}; cnt={}
    def mk(n,mod):
        def hook(_,inp,__):
            x=inp[0].detach().float()
            if isinstance(mod,nn.Conv2d):
                x=F.unfold(x,mod.kernel_size,padding=mod.padding,stride=mod.stride,dilation=mod.dilation).transpose(1,2)
            x=x.reshape(-1,x.shape[-1])
            if x.shape[0]>20000: x=x[torch.randperm(x.shape[0],device=x.device)[:20000]]
            if n not in H: H[n]=torch.zeros(x.shape[1],x.shape[1],device=dev); cnt[n]=0
            H[n]+=x.T@x; cnt[n]+=x.shape[0]
        return hook
    hs=[mod.register_forward_hook(mk(n,mod)) for n,mod in targets]
    with torch.no_grad():
        for i in range(0,len(Xc),32): fwd(prep(Xc[i:i+32]))
    for h in hs: h.remove()
    for n in H: H[n]/=cnt[n]
    # ---- quantizer
    def quantize(W,Hm,bits,g,mask,alpha=None):
        rows,cols=W.shape; W=W.clone(); qmax=2**(bits-1)-1; qmin=-2**(bits-1)
        if alpha: qmax,qmin=127,-127
        gg=cols if g<=0 else min(g,cols)
        if Hm is None: Hinv=torch.eye(cols,device=dev)
        else:
            Hm=Hm.clone(); dead=torch.diag(Hm)==0; Hm[dead,dead]=1; W[:,dead]=0
            Hm+=0.01*torch.diag(Hm).mean()*torch.eye(cols,device=dev)
            Hinv=torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(Hm)),upper=True)
        Q=torch.zeros_like(W); C=torch.zeros(rows,cols,dtype=torch.int8,device=dev)
        s=None; nscales=0
        for b0 in range(0,cols,128):
            b1=min(b0+128,cols); W1=W[:,b0:b1].clone(); E=torch.zeros_like(W1); Hi=Hinv[b0:b1,b0:b1]
            for j in range(b1-b0):
                c=b0+j
                if c%gg==0:
                    blk=W[:,c:c+gg] if j==0 else torch.cat([W1[:,j:],W[:,b1:]],1)[:,:min(gg,cols-c)]
                    if mask is not None: blk=blk*mask[:,c:c+gg]
                    mx=blk.abs().amax(1,keepdim=True).clamp(min=1e-8); best=None
                    if alpha: s=alpha*blk.pow(2).mean(1,keepdim=True).sqrt().clamp(min=1e-8)
                    for r in (() if alpha else (1.0,.9,.8,.7,.6,.5)):
                        sc=mx*r/max(qmax,1); q=(blk/sc).round().clamp(qmin,qmax); e=((q*sc-blk)**2).sum(1)
                        if best is None: best=e.clone(); s=sc.clone()
                        else:
                            k=e<best; best=torch.where(k,e,best); s=torch.where(k,sc,s)
                    nscales+=rows
                w=W1[:,j]; q=(w/s[:,0]).round().clamp(qmin,qmax)
                if mask is not None: q=q*mask[:,c]
                dq=q*s[:,0]; Q[:,c]=dq; C[:,c]=q.to(torch.int8)
                err=(w-dq)/Hi[j,j]; W1[:,j:]-=err[:,None]*Hi[j,j:][None,:]; E[:,j]=err
            if Hm is not None and b1<cols: W[:,b1:]-=E@Hinv[b0:b1,b1:]
        return Q,C,nscales
    def ent_bits(C):
        h=torch.bincount((C.flatten().long()+128),minlength=256).double(); p=h/h.sum(); p=p[p>0]
        return float(-(p*p.log2()).sum()*C.numel())
    def rel_err_rtn(W2,Hm,alpha,g):
        rows,cols=W2.shape; gg=cols if g<=0 else min(g,cols); pad=(-cols)%gg
        Wp=F.pad(W2,(0,pad)).reshape(rows,-1,gg); sc=alpha*Wp.pow(2).mean(2,keepdim=True).sqrt().clamp(min=1e-8)
        Qp=((Wp/sc).round().clamp(-127,127)*sc).reshape(rows,-1)[:,:cols]; D=W2-Qp
        return float(((D@Hm)*D).sum()/((W2@Hm)*W2).sum().clamp(min=1e-12))
    def pick_alpha(W2,Hm,e,g):
        lo,hi=0.02,3.0
        for _ in range(14):
            mid=(lo*hi)**0.5
            if rel_err_rtn(W2,Hm,mid,g)>e: hi=mid
            else: lo=mid
        return lo
    def apply(cfg):
        m.load_state_dict(base_sd); bits_ent=0.0; bits_fix=0.0; tq=0
        if cfg.get("ffn") or cfg.get("drop"): raise NotImplementedError
        if cfg.get("fp16only"):
            pass
        qn=set()
        for n,mod in targets:
            W=mod.weight.data.float(); rows=W.shape[0]; W2=W.reshape(rows,-1)
            if cfg.get("rank"):   # low-rank fp16 factors
                U,S,Vh=torch.linalg.svd(W2,full_matrices=False); r=max(1,int(cfg["rank"]*min(W2.shape)))
                Wr=(U[:,:r]*S[:r])@Vh[:r]; mod.weight.data=Wr.reshape(W.shape).half().float(); bits_ent+=r*sum(W2.shape)*16; bits_fix+=r*sum(W2.shape)*16; tq+=W2.numel(); qn.add(n); continue
            bits=cfg.get("bits",8); g=cfg.get("g",128); alpha=cfg.get("alpha")
            if cfg.get("e0"): alpha=pick_alpha(W2,H[n],cfg["e0"]*(W2.numel()/589824)**cfg.get("pw",1.0),g)
            sp=cfg.get("sp",0.0); mask=None
            if sp>0:
                if cfg.get("score","wanda")=="wanda":
                    an=H[n].diag().sqrt()[None,:]; sc=W2.abs()*an
                else: sc=W2.abs()
                k=int(W2.shape[1]*sp); thr=sc.kthvalue(max(k,1),dim=1,keepdim=True)[0]; mask=(sc>thr).float()
            Hm=None if cfg["method"]=="rtn" else H[n]
            if cfg["method"]=="rtn" and mask is not None: W2=W2*mask
            Q,C,ns=quantize(W2,Hm,bits,g,mask,alpha)
            mod.weight.data=Q.reshape(W.shape); 
            bits_ent+=ent_bits(C)+ns*16; bits_fix+=W2.numel()*(bits if not alpha else 8)+ns*16; tq+=W2.numel(); qn.add(n)
        # others -> fp16
        other=0
        for pn,p in m.named_parameters():
            mn=pn.rsplit(".",1)[0]
            if pn.endswith("weight") and mn in qn: continue
            p.data=p.data.half().float(); other+=p.numel()
        bits_ent+=other*16; bits_fix+=other*16
        return bits_ent/8, bits_fix/8
    for cfg in spec["cfgs"]:
        t=time.time()
        if cfg.get("base"): m.load_state_dict(base_sd); be=bf=nparams*4
        else: be,bf=apply(cfg)
        out=dict(cfg=cfg,ratio_ent=be/res["fp32_bytes"],ratio_fix=bf/res["fp32_bytes"])
        for sp_ in spec.get("splits",["dev"]):
            p=predict(X[sp_]); out[sp_]=dict(acc=float((p==Y[sp_]).float().mean()),agree=float((p==base_pred[sp_]).float().mean()))
        out["sec"]=time.time()-t; res["cfgs"].append(out); print(json.dumps(out),flush=True)
    return res

@app.local_entrypoint()
def main(spec:str, out:str):
    r=run.remote(json.loads(spec)); open(out,"w").write(json.dumps(r,indent=1)); 
    print(r["model"],r["params"],r["base_acc"])
    for c in r["cfgs"]: print(c["cfg"],"ent%.3f fix%.3f"%(c["ratio_ent"],c["ratio_fix"]),{k:v for k,v in c.items() if k in("dev","exam")})
