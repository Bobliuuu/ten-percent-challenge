import modal, json
app = modal.App("tenpct-trackI-emb")
img = (modal.Image.debian_slim().pip_install("torch","transformers","datasets","numpy","sentencepiece","protobuf")
       .add_local_file("dev2.json","/root/dev2.json").add_local_file("dev2_hard.json","/root/dev2_hard.json")
       .add_local_file("embq.py","/root/embq.py"))
@app.function(image=img, gpu="A100", timeout=3000)
def run(M="textattack/bert-base-uncased-SST-2"):
    import torch, numpy as np, sys, collections
    sys.path.insert(0,"/root"); import embq
    from datasets import load_dataset
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    tok=AutoTokenizer.from_pretrained(M); m=AutoModelForSequenceClassification.from_pretrained(M).cuda().eval()
    D=lambda f:[(x["text"],x["label"]) for x in json.load(open(f))]
    d2=D("/root/dev2.json")[512:]; dh=D("/root/dev2_hard.json")
    def P(data):
        o=[]
        with torch.no_grad():
            for i in range(0,len(data),64):
                b=tok([t for t,_ in data[i:i+64]],padding=True,truncation=True,max_length=128,return_tensors="pt").to("cuda")
                o.append(m(**b).logits.argmax(-1).cpu())
        return torch.cat(o)
    E0=m.get_input_embeddings().weight.data.clone(); V,H=E0.shape
    b2=P(d2); bh=P(dh)
    tr=list(load_dataset("nyu-mll/glue","sst2")["train"]["sentence"][:])
    cnt=collections.Counter()
    for e in tok(tr,truncation=True,max_length=128)["input_ids"]: cnt.update(e)
    freq=torch.zeros(V); 
    for k,v in cnt.items(): freq[k]=v
    for s in tok.all_special_ids: freq[s]=1e9
    res=[]
    for name,kw in embq.SCHEMES:
        E,nbytes=embq.quantize_embedding(E0,freq,**kw)
        m.get_input_embeddings().weight.data=E.to(E0.device)
        r=dict(name=name,emb_bytes=nbytes,emb_MB=nbytes/1e6,agree=float((P(d2)==b2).float().mean()),agree_hard=float((P(dh)==bh).float().mean()))
        print(r,flush=True); res.append(r)
    m.get_input_embeddings().weight.data=E0
    return dict(V=V,H=H,n_seen=int((freq>0).sum()),res=res)
@app.local_entrypoint()
def main(models:str="textattack/bert-base-uncased-SST-2"):
    out=list(run.map(models.split(",")))
    json.dump(out,open("emb_results.json","w"),indent=1)
    for o in out:
        for r in o["res"]: print(r)
