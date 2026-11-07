import sys,glob,json,torch,numpy as np
from safetensors import safe_open
d=sys.argv[1]; G=128
tot=0;stats={}
for f in sorted(glob.glob(d+"/*.safetensors")):
    with safe_open(f,"pt") as sf:
        for k in sf.keys():
            t=sf.get_tensor(k); n=t.numel(); tot+=n
            if t.ndim==2 and n>1e6:
                x=t.float().cuda()
                if x.shape[1]%G: print(k,"not divisible",x.shape);continue
                g=x.reshape(-1,G)
                mx=g.abs().amax(1,keepdim=True)
                # per group: values in {-s,0,+s}?
                s=mx
                r=torch.where(s>0,g/s,torch.zeros_like(g))
                tern=((r.abs()<1e-3)|((r.abs()-1).abs()<1e-3)).all(1)
                nz=(r.abs()<1e-3).float().mean().item()
                # exact check in fp16: g == sign*s or 0
                recon=torch.where(r.abs()>0.5,torch.sign(g)*s,torch.zeros_like(g))
                exact=(recon==g).all(1)
                # 1-bit: no zeros
                bin_=((g.abs()-s).abs()==0).all(1)
                stats[k]=(tuple(t.shape),str(t.dtype),exact.float().mean().item(),bin_.float().mean().item(),nz)
                del x,g
            else:
                stats[k]=(tuple(t.shape),str(t.dtype),None,None,None)
bad=[k for k,v in stats.items() if v[2] is not None and v[2]<1]
print("params",tot,"tensors",len(stats),"matrices non-exact:",len(bad))
for k,v in list(stats.items())[:6]+[(k,stats[k]) for k in bad[:5]]+[(k,v) for k,v in stats.items() if 'embed' in k or 'lm_head' in k or 'layers.0.' in k]: print(k,v)
