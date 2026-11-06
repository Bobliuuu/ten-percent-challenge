import torch, numpy as np, torch.nn.functional as F, copy, time
from lib import *
_vocab={}
def vocab_ids():
    if "v" not in _vocab:
        import os,json
        for pth in ("/root/vocab_ids.json","/Users/jhome/ten-percent/trackE/vocab_ids.json"):
            if os.path.exists(pth): _vocab["v"]=json.load(open(pth)); return _vocab["v"]
        d=data(); tok=d["tok"]
        ids=set(tok.all_special_ids)
        enc=tok(list(d["tr"]["sentence"]),truncation=True,max_length=64)["input_ids"]
        for e in enc: ids.update(e)
        _vocab["v"]=sorted(ids)
    return _vocab["v"]
@torch.no_grad()
def dense_targets(D,cb,srcs):
    T={s:([],[]) for s in set(srcs)}
    for ix,ids,m in cb:
        x=D.embed(ids)
        for i,L in enumerate(D.layers):
            c,z1,a,h,z2,x=D.layer_parts(L,x,m)
            if i in T:
                mb=m.bool(); T[i][0].append(z1[mb].half()); T[i][1].append(z2[mb].half())
    return {s:(torch.cat(a),torch.cat(b)) for s,(a,b) in T.items()}
@torch.no_grad()
def build(sd,plan,cb,log=print):
    """plan: layers=[(src,nh,nf)], vocab=bool, rank=int|None, lam=float, refit_full=bool"""
    D=Net(sd); P=Net(sd)
    lam=plan.get("lam",1e-3)
    if plan.get("vocab"):
        ids=vocab_ids(); vm=torch.full((30522,),100,dtype=torch.long); 
        for n,i in enumerate(ids): vm[i]=n
        P.vmap=vm.to(DEV); P.word=D.word[torch.tensor(ids,device=DEV)]
    r=plan.get("rank")
    if r:
        U,S,Vt=torch.linalg.svd(P.word.cpu().double(),full_matrices=False)
        P.word=(U[:,:r]*S[:r]).float().to(DEV); P.wfac=Vt[:r].float().to(DEV)
    tg=dense_targets(D,cb,[l[0] for l in plan["layers"]])
    xs=[P.embed(ids) for _,ids,_ in cb]; ms=[m for _,_,m in cb]
    P.layers=[]
    for (src,nh,nf) in plan["layers"]:
        L=copy.copy(D.layers[src]); L["src"]=src
        z1d,z2d=tg[src]
        # ---- attention heads
        Lf=dict(L)  # dense source layer weights on pruned-stream inputs
        feats=[];X=[]
        for x,m in zip(xs,ms):
            c=D.attn_ctx(Lf,x,m); feats.append(c[m.bool()]); X.append(x[m.bool()])
        C=torch.cat(feats); X=torch.cat(X)
        if nh<12:
            Wo=L["o"][0]
            imp=torch.stack([(C[:,h*64:(h+1)*64]@Wo[:,h*64:(h+1)*64].T).pow(2).sum(1).mean() for h in range(12)])
            kh=imp.topk(nh)[1].sort()[0].tolist()
            cols=torch.cat([torch.arange(h*64,(h+1)*64) for h in kh]).to(DEV) if nh else torch.zeros(0,dtype=torch.long,device=DEV)
            for key in "qkv": L[key]=(L[key][0][cols],L[key][1][cols])
            L["nh"]=nh
            if nh>0:
                Wn,bn=ridge(C[:,cols],z1d.float()-X,lam); L["o"]=(Wn,bn)
            else:
                bn=(z1d.float()-X).mean(0); L["o"]=(L["o"][0][:,:0],bn)
        elif plan.get("refit_full"):
            Wn,bn=ridge(C,z1d.float()-X,lam); L["o"]=(Wn,bn)
        del C,X,feats
        # ---- ffn
        As=[];Hs=[]
        for x,m in zip(xs,ms):
            z1=x+F.linear(D.attn_ctx(L,x,m),*L["o"]) if L["nh"]>0 else x+L["o"][1]
            a=F.layer_norm(z1,(768,),*L["ln1"],eps=1e-12)
            As.append(a[m.bool()]); Hs.append(F.gelu(F.linear(a,*L["w1"]))[m.bool()])
        A=torch.cat(As); H=torch.cat(Hs); del As,Hs
        if nf<3072:
            imp=H.pow(2).mean(0).sqrt()*L["w2"][0].norm(dim=0)
            kn=imp.topk(nf)[1].sort()[0]
            L["w1"]=(L["w1"][0][kn],L["w1"][1][kn]); Hk=H[:,kn]
            if nf>0:
                Wn,bn=ridge(Hk,z2d.float()-A,lam); L["w2"]=(Wn,bn)
            else:
                L["w2"]=(L["w2"][0][:,:0],(z2d.float()-A).mean(0))
        elif plan.get("refit_full"):
            Wn,bn=ridge(H,z2d.float()-A,lam); L["w2"]=(Wn,bn)
        del A,H
        P.layers.append(L)
        xs=[P.layer_parts(L,x,m)[-1] for x,m in zip(xs,ms)]
    return P
def agreement(P,base_logits,texts):
    p=predict(P,texts); return float((p.argmax(1)==base_logits.argmax(1)).mean())

def mk(nl,nh,nf,rank,lam=1e-3,vocab=True):
    src=sorted(set(np.linspace(0,11,nl).round().astype(int).tolist())) if nl<12 else list(range(12))
    return dict(layers=[(s,nh,nf) for s in src],vocab=vocab,rank=rank,lam=lam,tag=f"L{nl}h{nh}f{nf}r{rank}")
