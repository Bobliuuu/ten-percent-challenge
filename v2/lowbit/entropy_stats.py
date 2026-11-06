"""Ternary entropy stats for packed BitNet-style checkpoints. MODEL=... modal run --detach entropy_stats.py"""
import modal, json, os
app = modal.App("tenpct-lowbit-ent")
img = modal.Image.debian_slim().pip_install("torch","transformers==4.57.1","numpy","safetensors","huggingface_hub")
vol = modal.Volume.from_name("lowbit-data", create_if_missing=True)
hf = modal.Volume.from_name("hf-cache", create_if_missing=True)
@app.function(image=img, cpu=8, memory=32768, timeout=7200, volumes={"/hf":hf,"/d":vol})
def run(M):
    os.environ["HF_HOME"]="/hf"
    import torch, numpy as np, math
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open
    from transformers.integrations.bitnet import unpack_weights
    p=hf_hub_download(M,"model.safetensors")
    res={}; tot_n=0; tot_h0=0; tot_h1=0; tot_h2=0
    def H(c):
        c=c[c>0].astype(np.float64); p=c/c.sum(); return float(-(c*np.log2(p)).sum())
    with safe_open(p,"pt") as f:
        for k in f.keys():
            if not k.endswith("proj.weight"): continue
            u=f.get_tensor(k); w=unpack_weights(u,torch.bfloat16)  # shape [rows,cols]
            w=(w.to(torch.int8)).numpy(); n=w.size
            cnt=np.bincount((w+1).ravel(),minlength=3)
            h0=H(cnt)
            # context: left neighbour in row (cols), conditional entropy
            ctx=(w[:,:-1]+1)*3+(w[:,1:]+1); c2=np.bincount(ctx.ravel(),minlength=9).reshape(3,3)
            h1=sum(H(c2[i]) for i in range(3))+H(np.bincount((w[:,0]+1),minlength=3))
            # context: up neighbour
            ctxu=(w[:-1]+1)*3+(w[1:]+1); c3=np.bincount(ctxu.ravel(),minlength=9).reshape(3,3)
            h2=sum(H(c3[i]) for i in range(3))
            # per-row-adaptive static entropy (row prob tables, cost 3 params/row ignored here)
            hrow=sum(H(np.bincount(r+1,minlength=3)) for r in w)
            # per-column
            hcol=sum(H(np.bincount(r+1,minlength=3)) for r in w.T)
            res[k]=dict(n=n,frac=(cnt/n).tolist(),h0=h0/n,h_left=h1/n,h_up=h2/n,h_row=hrow/n,h_col=hcol/n)
            tot_n+=n; tot_h0+=h0; tot_h1+=min(h1,h2)
            print(k,res[k],flush=True)
    out=dict(model=M,total_n=tot_n,bits_per_w_h0=tot_h0/tot_n,bits_per_w_ctx=tot_h1/tot_n,
      h0_bytes=tot_h0/8,base3_bytes=tot_n*math.log2(3)/8,per={k:v for k,v in res.items()})
    os.makedirs("/d/ent",exist_ok=True); json.dump(out,open("/d/ent/"+M.replace("/","_")+".json","w")); vol.commit()
    print("RESULT_JSON "+json.dumps({k:v for k,v in out.items() if k!="per"})); return out
@app.local_entrypoint()
def main():
    for m in os.environ["MODELS"].split(","): run.spawn(m)
