import sys,glob,os,json,torch
from safetensors.torch import load_file,save_file
src=glob.glob(sys.argv[1])[0]; dst=sys.argv[2]; variant=sys.argv[3]  # variant: bin_both | bin_embed | bin_head
os.makedirs(dst,exist_ok=True); G=128
for f in os.listdir(src):
    p=os.path.join(src,f)
    if f.endswith(".safetensors"): continue
    if not os.path.exists(os.path.join(dst,f)): os.symlink(os.path.realpath(p),os.path.join(dst,f))
def binarize(w):
    x=w.float().reshape(-1,G); sc=x.abs().mean(1,keepdim=True)
    # keep exact when group already 1-bit; for ternary: sign with zeros -> use +sc (sign(0)=>+)
    q=torch.where(x>=0,sc,-sc); return q.reshape(w.shape).to(w.dtype)
for f in sorted(glob.glob(src+"/*.safetensors")):
    d=load_file(f); ch=False
    for k in list(d):
        if (k=="model.embed_tokens.weight" and variant in("bin_both","bin_embed")) or (k=="lm_head.weight" and variant in("bin_both","bin_head")):
            w=d[k]; q=binarize(w); print(k,"rel_mse",(((q.float()-w.float())**2).sum()/(w.float()**2).sum()).item()); d[k]=q; ch=True
    if ch: save_file(d,os.path.join(dst,os.path.basename(f)),metadata={"format":"pt"})
    else:
        t=os.path.join(dst,os.path.basename(f)); 
        if not os.path.exists(t): os.symlink(os.path.realpath(f),t)
