import sys,glob,torch,json,collections
from safetensors import safe_open
def Hc(c):
    c=c.double(); p=c[c>0]/c.sum(); return -(p*p.log2()).sum().item()
tern_bits=0.0; tern_n=0; scale_n=0; raw_bytes=0; raw_by=collections.Counter(); total_n=0
for f in sorted(glob.glob(sys.argv[1]+"/*.safetensors")):
    with safe_open(f,"pt") as sf:
        for k in sf.keys():
            t=sf.get_tensor(k); total_n+=t.numel()
            if t.ndim<2 or t.numel()<1e5: raw_bytes+=t.numel()*2; raw_by["small"]+=t.numel()*2; continue
            x=t.float().reshape(t.shape[0],-1)
            ok=True
            for row in x[:: max(1,x.shape[0]//8)][:8]:
                if len(torch.unique(row))>3: ok=False;break
            if not ok: raw_bytes+=t.numel()*2; raw_by[k.split(".")[-2] if "layers" in k else k]+=t.numel()*2; continue
            sg=(torch.sign(x)+1).long(); c=torch.bincount(sg.flatten(),minlength=3)
            tern_bits+=Hc(c)*x.numel(); tern_n+=x.numel(); scale_n+=x.shape[0]
print("total params",total_n,"fp16 bytes",total_n*2)
print("ternary params",tern_n,"trit bpw %.4f"%(tern_bits/tern_n),"trit MB",tern_bits/8e6,"scales(fp16, raw) MB",scale_n*2/1e6)
print("raw non-ternary MB",raw_bytes/1e6,dict((k,round(v/1e6,1)) for k,v in raw_by.most_common(8)))
tot=tern_bits/8+scale_n*2+raw_bytes
print("TOTAL MB %.1f  = %.2f%% of fp16"%(tot/1e6,100*tot/(total_n*2)))
