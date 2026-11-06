"""Binoculars-style zero-shot AI-text score for suite B (dev AUROC; threshold fit on calib). No training of any weights. Deploy: modal deploy bino.py; python3 spawn_bino.py"""
import modal, json, os
app = modal.App("tenpct-commercial-bino")
img = (modal.Image.debian_slim().pip_install("torch","transformers==4.51.3","numpy","accelerate","scikit-learn")
       .add_local_file("B/dev.jsonl","/root/B_dev.jsonl").add_local_file("B/calib.jsonl","/root/B_calib.jsonl"))
hf = modal.Volume.from_name("hf-cache", create_if_missing=True)
@app.function(image=img, gpu=["A100-40GB","L40S"], timeout=7200, volumes={"/hf":hf})
def run(size="1.5B", n=1000):
    import torch, numpy as np, time
    os.environ["HF_HOME"]="/hf"
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from sklearn.metrics import roc_auc_score
    obs_n,perf_n=f"Qwen/Qwen2.5-{size}",f"Qwen/Qwen2.5-{size}-Instruct"
    tok=AutoTokenizer.from_pretrained(obs_n)
    o=AutoModelForCausalLM.from_pretrained(obs_n,torch_dtype=torch.bfloat16).cuda().eval()
    p=AutoModelForCausalLM.from_pretrained(perf_n,torch_dtype=torch.bfloat16).cuda().eval()
    def score(txt):
        x=torch.tensor([tok.encode(txt)[:512]]).cuda()
        with torch.no_grad():
            lo=o(x).logits[0,:-1].float(); lp=p(x).logits[0,:-1].float(); t=x[0,1:]
        ppl=torch.nn.functional.cross_entropy(lp,t)           # performer log-ppl
        xppl=-(torch.softmax(lo,-1)*torch.log_softmax(lp,-1)).sum(-1).mean()  # cross-entropy observer->performer
        return float(ppl/xppl), float(ppl)
    def load(f,k): 
        r=[json.loads(l) for l in open(f)][:k]; return r
    cal=load("/root/B_calib.jsonl",600); dev=load("/root/B_dev.jsonl",n)
    sc=np.array([score(r["text"]) for r in cal]); yc=np.array([r["answer"] for r in cal])
    sd=np.array([score(r["text"]) for r in dev]); yd=np.array([r["answer"] for r in dev])
    res={"size":size,"n_dev":len(dev)}
    for j,name in [(0,"binoculars"),(1,"logppl")]:
        # AI text => lower score; orient
        auc=roc_auc_score(yd,-sd[:,j]); ths=np.unique(sc[:,j]); best=max(ths,key=lambda t:((sc[:,j]<t)==yc).mean())
        res[name]=dict(auroc=float(auc),acc_at_calib_thr=float(((sd[:,j]<best)==yd).mean()),calib_acc=float(((sc[:,j]<best)==yc).mean()),thr=float(best))
    import collections
    dom=collections.defaultdict(list)
    for r,s in zip(dev,sd[:,0]): dom[r["src"].split("_")[0]].append((r["answer"],-s))
    res["per_domain_auroc_bino"]={k:float(roc_auc_score([a for a,_ in v],[b for _,b in v])) for k,v in dom.items() if len(set(a for a,_ in v))==2}
    print("RESULT_JSON "+json.dumps(res)); return res
