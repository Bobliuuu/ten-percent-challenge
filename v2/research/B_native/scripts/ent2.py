import sys,glob,torch,math,json
from safetensors import safe_open
G=128
def Hc(x,n):
    c=torch.bincount(x.flatten().long(),minlength=n).double(); p=c[c>0]/c.sum(); return -(p*p.log2()).sum().item()
acc={}
for f in sorted(glob.glob(sys.argv[1]+"/*.safetensors")):
    with safe_open(f,"pt") as sf:
        for k in sf.keys():
            t=sf.get_tensor(k)
            cls="embed" if "embed" in k else "head" if "lm_head" in k else "body" if (t.ndim==2 and t.numel()>1e6) else "small"
            a=acc.setdefault(cls,dict(n=0,trit_bits=0.0,scale_bits=0.0,groups=0,small_bytes=0))
            if cls=="small": a["small_bytes"]+=t.numel()*2; continue
            g=t.reshape(-1,G); s=g.abs().amax(1); tr=(torch.sign(g.float())+1).to(torch.uint8)
            # per-row-of-groups adaptive: trit entropy conditioned on group zero-count class (0..128 -> 8 buckets)
            z=(tr==1).sum(1); 
            a["n"]+=g.numel(); a["groups"]+=g.shape[0]
            a["trit_bits"]+=Hc(tr,3)*g.numel()
            sb=s.view(torch.int16).long()&0xffff
            a["scale_bits"]+=Hc(sb,65536)*sb.numel()
for c,a in acc.items():
    if a["n"]: print(c,"n",a["n"],"trit bpw %.4f"%(a["trit_bits"]/a["n"]),"scale bits/group %.2f"%(a["scale_bits"]/a["groups"]),"MB trit %.1f scale %.1f"%(a["trit_bits"]/8e6,a["scale_bits"]/8e6))
    else: print(c,a["small_bytes"])
tot=sum((a["trit_bits"]+a["scale_bits"])/8 for a in acc.values() if a["n"])+acc["small"]["small_bytes"]
print("TOTAL bytes (ideal entropy, excl tokenizer/code)",tot, "fp16 bytes", 2*sum(a["n"] for a in acc.values() if a["n"])+acc["small"]["small_bytes"])
