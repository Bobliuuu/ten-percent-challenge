"""Sequential GPTQ with a per-tensor config -> fake-quant model -> TPK2 container. Used locally or inside Modal (mbuild.py).
cfg: dict(enc={'L0.attention.self.query':[bits,g],...}, emb=[[bits,g] per freq group], pool=[bits,g], damp, order('none'|'act'), clip, calib('dev'|'cal'|'mix'), beta(None=grid))"""
from core import *
import fmt
def sequential_quant(m,cfg,Bcal):
    m=m.to(DEV).eval(); qd={}
    damp=cfg.get("damp",0.01); order=cfg.get("order","none")
    for l in range(12):
        Hs=collect_H_layer(m,Bcal,l)
        for n in LIN:
            key=f"L{l}.{n}"; b,g=cfg["enc"][key]; mod=m.bert.encoder.layer[l].get_submodule(n)
            W=mod.weight.data; H=Hs[key].to(W.device)
            od=None
            if order=="act": od=torch.argsort(torch.diag(H),descending=True)
            if cfg.get("rot",1):
                R=rot_mat(W.shape[1],f"bert.encoder.layer.{l}.{n}.weight").to(W.device); q,s,z=gptq2(W@R,R.T@H@R,b,g,damp=damp,order=od,clip=cfg.get("clip",True))
                mod.weight.data=(dequant(q.cpu(),s.cpu(),z.cpu(),g).to(DEV))@R.T
            else:
                q,s,z=gptq2(W,H,b,g,damp=damp,order=od,clip=cfg.get("clip",True)); mod.weight.data=dequant(q.cpu(),s.cpu(),z.cpu(),g).to(DEV)
            qd[f"bert.encoder.layer.{l}.{n}.weight"]=(q.cpu().float(),s.cpu(),z.cpu(),b,g)
    groups=emb_tiers(); E=m.bert.embeddings.word_embeddings; W0=E.weight.data.cpu().clone(); W=W0.clone(); gid=np.zeros(W0.shape[0],np.uint8); mu=W0.mean(0).half().float()
    for j,idx in enumerate(groups):
        b,g=cfg["emb"][j]; q,s,lv=quant_lm(W0[idx],b,g,mu); W[idx]=dequant_lm(q,s,lv,g,mu); gid[idx.numpy()]=j
        qd[f"bert.embeddings.word_embeddings.weight#{j}"]=(q,s,None,b,g,lv.numpy())
    qd["_mu"]=mu
    E.weight.data=W.to(DEV)
    b,g=cfg["pool"]; P=m.bert.pooler.dense; q,s,z=quant_clip(P.weight.data.cpu(),b,g,fs=(1.0,.9,.8)); P.weight.data=dequant(q,s,z,g).to(DEV)
    qd["bert.pooler.dense.weight"]=(q,s,z,b,g)
    return m,qd,gid
@torch.no_grad()
def collect_H_layer(m,batches,l):
    Hs={}; cur={}; hs=[]; cnt=0; L=m.bert.encoder.layer[l]
    def mk(key):
        def hook(mod,inp):
            x=inp[0].float()[cur["m"].bool()]; Hs[key]=Hs.get(key,0)+(x.T@x)
        return hook
    for n in LIN: hs.append(L.get_submodule(n).register_forward_pre_hook(mk(f"L{l}.{n}")))
    for ids,tt,am in batches:
        cur["m"]=am.to(DEV); cnt+=int(am.sum()); m(input_ids=ids.to(DEV),token_type_ids=tt.to(DEV),attention_mask=am.to(DEV))
    for h in hs: h.remove()
    return {k:(2*v/cnt) for k,v in Hs.items()}
def small_dict(m,qd):
    out={}
    for n,p in m.state_dict().items():
        if "position_ids" in n or n.endswith("token_type_ids"): continue
        if n=="bert.embeddings.word_embeddings.weight" or n in qd: continue
        out[n]=p.detach().cpu().half().numpy()
    return out
def model_from(cfgd,W):
    from transformers import BertConfig, AutoModelForSequenceClassification
    c=BertConfig.from_dict(cfgd); c.max_position_embeddings=W['bert.embeddings.position_embeddings.weight'].shape[0]; m=AutoModelForSequenceClassification.from_config(c).eval(); r=m.load_state_dict(W,strict=False); assert not r.missing_keys and not r.unexpected_keys,(r.missing_keys,r.unexpected_keys); return m
def build(cfg,out,log=print):
    D=data(); calib={"dev":D["B_dev"],"cal":D["B_cal"],"mix":D["B_dev"]+D["B_cal"]}[cfg.get("calib","dev")]
    base=fullbase().to(DEV); bev=logits(base,D["B_ev"]); bh=logits(base,D["B_h"]); bcal=logits(base,D["B_cal"]); bdev=logits(base,D["B_dev"])
    m,qd,gid=sequential_quant(fullbase(),cfg,calib)
    # bias fold into classifier.bias (zero extra bytes)
    cl=logits(m,D["B_cal"]); d0=(cl[:,1]-cl[:,0])
    beta=cfg.get("beta")
    if beta is None:
        best=(-1,0)
        for bt in np.arange(-0.5,1.51,0.125):
            a=(((d0+bt)>0)==((bcal[:,1]-bcal[:,0])>0)).float().mean().item()
            if a>best[0]+1e-9: best=(a,float(bt))
        beta=best[1]
    cb=m.classifier.bias.data.cpu(); cb=torch.tensor([cb[0]-beta/2,cb[1]+beta/2]); m=m.cpu(); m.classifier.bias.data=cb.half().float()
    mu=qd.pop("_mu"); sd=small_dict(m,qd); sd["emb_mu"]=mu.half().numpy(); vt=[tok.convert_ids_to_tokens(i) for i in range(tok.vocab_size)]
    size=fmt.export(out,qd,sd,gid,vt,m.config.to_dict())
    cfgd,W,vocab=fmt.read(out); m2=model_from(cfgd,W)
    res=dict(size=size,beta=beta)
    for nm,Bs,bl in [("ev",D["B_ev"],bev),("h",D["B_h"],bh),("cal",D["B_cal"],bcal)]:
        l2=logits(m2,Bs); kl,ag=kl_agree(l2,bl); res[nm+"_agree"]=ag; res[nm+"_kl"]=kl
    l1=logits(m,D["B_ev"]); res["reload_maxdiff"]=float((l1-logits(m2,D["B_ev"])).abs().max())
    lb=logits(m2,D["B_dev"]); res["sstdev_agree"]=(lb.argmax(-1)==bdev.argmax(-1)).float().mean().item(); res["sstdev_bias"]=float(((lb[:,1]-lb[:,0])-(bdev[:,1]-bdev[:,0])).mean())
    return res
