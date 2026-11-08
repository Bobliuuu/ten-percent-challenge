import json,sys
R=0.0963; base=dict(target_ratio=R,emb_L=4,head_L=4,g=256)
V=dict(vk=65536,emb_trim=1,head_trim=1)
def cf(p,name,**kw): return dict(name=f"{p}_{name}",**base,**kw)
def arms(p):
    A=[cf(p,"b096"),
       cf(p,"v64",**V),
       cf(p,"f90",ffn_keep=0.9),cf(p,"f75",ffn_keep=0.75),cf(p,"f60",ffn_keep=0.6),
       cf(p,"d2",drop=2),cf(p,"d4",drop=4),
       cf(p,"k75",kv_keep=0.75),
       cf(p,"v64f90",ffn_keep=0.9,**V),cf(p,"v64f75",ffn_keep=0.75,**V),
       cf(p,"f90flap",ffn_keep=0.9,crit="flap"),cf(p,"f90mix",ffn_keep=0.9,mixneutral=300),
       cf(p,"v64f75d2k75",ffn_keep=0.75,drop=2,kv_keep=0.75,**V)]
    return A
def pr(p):  # pruning-only damage (near-lossless 4-bit everywhere)
    nb=dict(L=16,emb_L=16,head_L=16,g=256,target_ratio=1.0)
    return [dict(name=f"{p}_u_f75",ffn_keep=0.75,**nb),dict(name=f"{p}_u_f60",ffn_keep=0.6,**nb),dict(name=f"{p}_u_d4",drop=4,**nb),dict(name=f"{p}_u_k75",kv_keep=0.75,**nb),dict(name=f"{p}_u_v64",**V,**nb)]
model=sys.argv[1]; flag={"i":"injr","n":"news"}
jobs=[]
for p in "in":
    a=arms(p)+pr(p)
    chunks=[a[:5],a[5:10],a[10:]]
    for k,c in enumerate(chunks):
        jobs.append(dict(model=model,tag=f"p14_{p}{k}",size="s",**{flag[p]:True},cfgs=c))
json.dump(jobs,open(sys.argv[2],"w"),indent=1); print(len(jobs),sum(len(j['cfgs']) for j in jobs))
