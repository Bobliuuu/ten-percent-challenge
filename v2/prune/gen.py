import json,sys
def w(name,cs): json.dump(cs,open(name,"w"),indent=1); print(name,len(cs))
# Phase 1: removal only (no quant)
p1=[]
for n in [2,4,6,8,10,12,16]:
    for pt in ["none","256","full"]: p1.append(dict(name=f"p1_drop{n}_{pt}",drop=n,patch=pt))
for k in [0.9,0.75,0.6,0.5,0.4,0.3]:
    p1.append(dict(name=f"p1_ffn{k}_rec",ffn_keep=k,ffn_rec=True)); p1.append(dict(name=f"p1_ffn{k}_norec",ffn_keep=k,ffn_rec=False))
for k in [0.875,0.75,0.5]:
    p1.append(dict(name=f"p1_head{k}_rec",head_keep=k,head_rec=True)); p1.append(dict(name=f"p1_head{k}_norec",head_keep=k,head_rec=False))
p1.append(dict(name="p1_dense16"))
for src,dst in [(20,21),(30,31),(20,22)]: p1.append(dict(name=f"p1_tie{src}_{dst}",tie=[[src,dst]]))
p1.append(dict(name="p1_delta",mode="delta"))
w("p1.json",p1)
# Phase 2: dense GPTQ baselines + lowrank
p2=[]
for b in [2,3,4]:
    for g in [128,64]: p2.append(dict(name=f"p2_gptq{b}_g{g}",bits=b,g=g))
for b,r in [(2,64),(2,128),(3,32),(3,64)]: p2.append(dict(name=f"p2_gptq{b}_lr{r}",bits=b,g=128,lr_rank=r))
w("p2.json",p2)
