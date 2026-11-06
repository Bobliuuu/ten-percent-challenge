import modal,json,sys,os
R=os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),'..'))
app=modal.App("tp-sens")
img=(modal.Image.debian_slim().pip_install("torch","transformers","datasets","numpy")
 .add_local_file(R+"/splits.json","/root/tp/splits.json").add_local_file(R+"/trackA/lib.py","/root/tp/trackA/lib.py")
 .add_local_file(R+"/trackG/dev2.py","/root/tp/trackG/dev2.py").add_local_file(R+"/trackG/dev2.json","/root/tp/trackG/dev2.json").add_local_file(R+"/trackG/dev2_hard.json","/root/tp/trackG/dev2_hard.json")
 .add_local_file(R+"/trackP/core.py","/root/tp/trackP/core.py"))
@app.function(image=img,gpu="L4",timeout=3000,max_containers=40)
def run(task):
    import sys; sys.path.insert(0,"/root/tp/trackP"); os.chdir("/root/tp/trackP")
    import importlib; C=importlib.import_module('core'); globals().update({k:getattr(C,k) for k in dir(C) if not k.startswith('_')})
    D=data(); m=fullbase().to(DEV)
    bl=logits(m,D["B_cal"]); res={}
    if task["kind"]=="enc":
        l=task["l"]; Hs=collect_H(m,D["B_dev"])
        for n in LIN:
            mod=m.bert.encoder.layer[l].get_submodule(n); W0=mod.weight.data.clone(); key=f"L{l}.{n}"
            for b in [2,3,4]:
                for g in [64,128,256]:
                    q,s,z=gptq2(W0,Hs[key],b,g)
                    mod.weight.data=dequant(q.cpu(),s.cpu(),z.cpu(),g).to(DEV)
                    kl,ag=kl_agree(logits(m,D["B_cal"]),bl); res[f"{key}|{b}|{g}"]=dict(kl=kl,ag=ag,size=est_size(q.cpu(),z.cpu(),b))
            mod.weight.data=W0
    else:  # emb groups + pooler (RTN w/ clip)
        groups,cnt=emb_groups(); sp=torch.tensor([0,100,101,102,103]); groups=[g_[~torch.isin(g_,sp)] for g_ in groups]; groups[-1]=torch.cat([groups[-1],sp])
        E=m.bert.embeddings.word_embeddings; W0=E.weight.data.clone()
        for gi,idx in enumerate(groups):
            for b in [1,2,3,4,6]:
                for g in [64,128,256]:
                    q,s,z=quant_clip(W0[idx].cpu(),b,g); W=W0.clone(); W[idx]=dequant(q,s,z,g).to(DEV); E.weight.data=W
                    kl,ag=kl_agree(logits(m,D["B_cal"]),bl); res[f"emb{gi}|{b}|{g}"]=dict(kl=kl,ag=ag,size=est_size(q,z,b),n=len(idx))
            E.weight.data=W0
        P=m.bert.pooler.dense; W0=P.weight.data.clone()
        for b in [3,4,5,6,8]:
            for g in [64,128]:
                q,s,z=quant_clip(W0.cpu(),b,g,fs=(1.0,.9,.8)); P.weight.data=dequant(q,s,z,g).to(DEV)
                kl,ag=kl_agree(logits(m,D["B_cal"]),bl); res[f"pool|{b}|{g}"]=dict(kl=kl,ag=ag,size=est_size(q,z,b))
    return res
@app.local_entrypoint()
def main():
    tasks=[dict(kind="enc",l=l) for l in range(12)]+[dict(kind="emb")]
    out={}
    for r in run.map(tasks): out.update(r)
    json.dump(out,open("sens.json","w")); print(len(out))
