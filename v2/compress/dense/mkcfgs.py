import json,sys
def base(model,size): return dict(model=model,size=size)
CFGS=[ # name, kwargs
 ("t096",dict(target_ratio=0.096,emb_L=4,head_L=4,g=256)),
 ("t096_dz15",dict(target_ratio=0.096,emb_L=4,head_L=4,g=256,dz=0.15)),
 ("t096_g1",dict(target_ratio=0.096,emb_L=4,head_L=4,g=256,gamma=1.0)),
 ("t096_drop4",dict(target_ratio=0.096,emb_L=4,head_L=4,g=256,drop=4)),
 ("t096_drop8",dict(target_ratio=0.096,emb_L=4,head_L=4,g=256,drop=8)),
 ("t096_norot",dict(target_ratio=0.096,emb_L=4,head_L=4,g=256,rot=False)),
 ("t15",dict(target_ratio=0.15,emb_L=6,head_L=6,g=256)),
 ("t20",dict(target_ratio=0.20,emb_L=8,head_L=8,g=128)),
 ("t25",dict(target_ratio=0.25,emb_L=16,head_L=16,g=128)),
 ("t30",dict(target_ratio=0.30,emb_L=16,head_L=16,g=128)),
]
def mk(model,size,quick=False,chunks=1,prefix=""):
    out=[]
    cs=[]
    for n,kw in CFGS:
        c=dict(model=model,name=prefix+n,**kw)
        if quick: c.update(qstride=4,qoff=0,nev=16,ntf=40,max_new=16,name=prefix+"Q_"+n)
        cs.append(c)
    k=(len(cs)+chunks-1)//chunks
    for i in range(0,len(cs),k): out.append(dict(model=model,size=size,multi=cs[i:i+k],name=f"multi{i}"))
    return out
L=[]
L+=mk("Qwen/Qwen3-8B","x",quick=True,chunks=1)+mk("Qwen/Qwen3-8B","x",chunks=3)
L+=mk("Qwen/Qwen3-14B","x",quick=True,chunks=1)+mk("Qwen/Qwen3-14B","x",chunks=4)
json.dump(L,open("sweep1.json","w")); print(len(L),[len(c["multi"]) for c in L])
