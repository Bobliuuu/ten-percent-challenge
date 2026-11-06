import sys, json, time, torch, numpy as np
from lib import *
def run(cfg):
    torch.manual_seed(0)
    sys.path.insert(0,'/Users/jhome/ten-percent/trackG'); from dev2 import load_dev2
    t2,y2=load_dev2(); th,yh=load_dev2(hard=True); dev,fb=data(); od=dev["sentence"]
    cal=t2[:cfg.get("ncal",512)]; ev=t2[512:]; y2=torch.tensor(y2[512:])
    m=load(); base_ev=preds(m,ev).argmax(-1); base_h=preds(m,th).argmax(-1); base_o=preds(m,od).argmax(-1)
    bits=cfg["bits"]; g=cfg["g"]; method=cfg["method"]
    bitsfor=cfg.get("bitsfor",lambda name:bits)
    tot_fixed=0; tot_ent=0; tot_meta=0
    def account(codes,s,z,b,numel):
        nonlocal tot_fixed,tot_ent,tot_meta
        tot_fixed+=numel*b; tot_ent+=entropy_bits(codes,b); tot_meta+=s.numel()*(16+b)
    layers=[(f"l{i}.{n}",getattr(getattr(l,a),b) if b else getattr(l,a)) for i,l in enumerate(m.bert.encoder.layer) for a,b,n in
            [("attention","self","q"),("attention","self","k"),("attention","self","v")] ] if False else None
    lst=[]
    for i,l in enumerate(m.bert.encoder.layer):
        lst.append([(f"l{i}.q",l.attention.self.query),(f"l{i}.k",l.attention.self.key),(f"l{i}.v",l.attention.self.value),
                    (f"l{i}.o",l.attention.output.dense),(f"l{i}.f1",l.intermediate.dense),(f"l{i}.f2",l.output.dense)])
    lst.append([("pool",m.bert.pooler.dense)])
    for grp in lst:
        # sequential: hessians from partially quantized model. attn/ffn inside a layer: do q,k,v | o | f1 | f2 in order for exactness
        stages=[grp[:3],grp[3:4],grp[4:5],grp[5:6]] if len(grp)==6 else [grp]
        for st in stages:
            Hs={n:torch.zeros(mod.in_features,mod.in_features) for n,mod in st}
            hooks=[]
            if method=="gptq":
                def mk(n):
                    def h(mod,inp,out):
                        x=inp[0].detach().float().reshape(-1,inp[0].shape[-1]).cpu()
                        mask=h.mask; x=x[mask] if (mask is not None and mask.numel()==x.shape[0]) else x
                        Hs[n].add_(x.T@x)
                    return h
                for n,mod in st: hk=mk(n); hk.mask=None; hooks.append(mod.register_forward_hook(hk))
                mm=m.to(dev_dev)
                with torch.no_grad():
                    for b in batches(cal,64):
                        b={k:v.to(dev_dev) for k,v in b.items()}
                        # mask padding tokens
                        flat=b["attention_mask"].reshape(-1).bool().cpu()
                        for hk in hooks: pass
                        for (n,mod),hh in zip(st,hooks):
                            pass
                        # set mask for each hook via closure attr
                        for (n,mod) in st:
                            for hobj in mod._forward_hooks.values(): hobj.mask=flat if mod.in_features and True else None
                        mm(**b)
                m.cpu()
                for hk in hooks: hk.remove()
            for n,mod in st:
                W=mod.weight.data; b_=bitsfor(n)
                if method=="rtn": d,codes,s,z=rtn(W,b_,g)
                else: d,codes,s,z=gptq(W,Hs[n],b_,g,actorder=cfg.get("actorder",True))
                mod.weight.data=d.float(); account(codes,s,z,b_,W.numel())
                if cfg.get("actorder",True) and method=="gptq": tot_meta+=W.shape[1]*16
    # embeddings
    eb=cfg.get("ebits",4); eg=cfg.get("eg",64)
    for name,emb in [("word",m.bert.embeddings.word_embeddings)]:
        if eb<16:
            d,codes,s,z=rtn(emb.weight.data,eb,eg); emb.weight.data=d; account(codes,s,z,eb,emb.weight.numel())
        else: tot_fixed+=emb.weight.numel()*16; tot_ent+=emb.weight.numel()*16
    # fp16 rest: position, type emb, LN, biases, classifier
    rest=0
    for n,p in m.named_parameters():
        if "word_embeddings" in n or (p.dim()==2 and any(k in n for k in ["query","key","value","attention.output.dense","intermediate.dense","output.dense","pooler"]) and "LayerNorm" not in n): continue
        rest+=p.numel(); p.data=p.data.half().float()
    fixed_bytes=(tot_fixed+tot_meta+rest*16)/8; ent_bytes=(tot_ent+tot_meta+rest*16)/8
    pe=preds(m,ev).argmax(-1); ph=preds(m,th).argmax(-1); po=preds(m,od).argmax(-1)
    r=dict(name=cfg["name"],bytes_fixed=fixed_bytes,bytes_ent=ent_bytes,ratio_fixed=fixed_bytes/BASE_BYTES,ratio_ent=ent_bytes/BASE_BYTES,
           acc2=float((pe==y2).float().mean()),base_acc2=float((base_ev==y2).float().mean()),agree2=float((pe==base_ev).float().mean()),agree_hard=float((ph==base_h).float().mean()),agree_olddev=float((po==base_o).float().mean()))
    return r
if __name__=="__main__":
    cfgs=json.loads(sys.argv[1])
    for c in cfgs:
        t=time.time(); r=run(c); r["sec"]=time.time()-t; print(json.dumps(r),flush=True)
