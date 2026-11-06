import modal, itertools, json
app = modal.App("tenpct-sweep")
img = (modal.Image.debian_slim().pip_install("torch","transformers","datasets","numpy")
       .add_local_file("splits.json","/root/splits.json").add_local_file("dev2.json","/root/dev2.json"))
M="textattack/bert-base-uncased-SST-2"
BASE_BYTES=109483778*4

@app.function(image=img, gpu="L4", timeout=1800, max_containers=100)
def run(cfg):
    import torch, copy, numpy as np
    from datasets import load_dataset
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    dev_idx=json.load(open("/root/splits.json"))["dev"]
    ds=load_dataset("nyu-mll/glue","sst2")  # only TRAIN is ever touched here
    tr=ds["train"]; d0=tr.select(dev_idx)
    D2=json.load(open("/root/dev2.json")); D2=D2 if isinstance(D2,list) else D2["data"]
    d={"sentence":[x["text"] for x in D2],"label":[x["label"] for x in D2]}
    tok=AutoTokenizer.from_pretrained(M)
    m=AutoModelForSequenceClassification.from_pretrained(M).cuda().eval()
    def preds(model):
        out=[]
        with torch.no_grad():
            for i in range(0,len(d["sentence"]),128):
                b=tok(d["sentence"][i:i+128],padding=True,truncation=True,max_length=64,return_tensors="pt").to("cuda")
                out+=model(**b).logits.argmax(-1).tolist()
        return np.array(out)
    y=np.array(d["label"]); base=preds(m)
    def fq(w,bits,g):
        if bits>=16: return w
        sh=w.shape; w=w.reshape(-1,g) if w.numel()%g==0 else w.reshape(1,-1)
        lo=w.min(1,keepdim=True)[0]; hi=w.max(1,keepdim=True)[0]
        n=2**bits-1; s=(hi-lo).clamp(min=1e-8)/n
        return (((w-lo)/s).round().clamp(0,n)*s+lo).reshape(sh)
    L=m.bert.encoder.layer
    keep=cfg["keep"]; fr=cfg["ffn"]; bits=cfg["bits"]; g=cfg["g"]; ebits=cfg["ebits"]
    m.bert.encoder.layer=torch.nn.ModuleList([L[i] for i in keep])
    nbits=0; n_enc=0
    for l in m.bert.encoder.layer:
        if fr<1.0:
            W1=l.intermediate.dense; W2=l.output.dense
            imp=W1.weight.norm(dim=1)*W2.weight.norm(dim=0)
            k=int(3072*fr); idx=imp.topk(k)[1].sort()[0]
            W1.weight.data=W1.weight.data[idx]; W1.bias.data=W1.bias.data[idx]; W1.out_features=k
            W2.weight.data=W2.weight.data[:,idx]; W2.in_features=k
        for mod in [l.attention.self.query,l.attention.self.key,l.attention.self.value,l.attention.output.dense,l.intermediate.dense,l.output.dense]:
            w=mod.weight.data; n=w.numel(); n_enc+=n
            mod.weight.data=fq(w,bits,g); nbits+=n*bits+(n//g)*32
            nbits+=mod.bias.numel()*16
        nbits+=4*768*16
    # vocab prune: tokens in TRAIN text only
    ids=set(tok.all_special_ids)
    for t in tr["sentence"]: ids.update(tok(t,truncation=True,max_length=64)["input_ids"])
    ids=sorted(ids) if cfg.get("vp",1) else list(range(30522)); emb=m.bert.embeddings.word_embeddings.weight.data
    keepmask=torch.zeros(emb.shape[0],dtype=torch.bool); keepmask[ids]=True
    e=emb.clone(); e[~keepmask.cuda()]=0  # simulate: unseen tokens -> zero (they'd map to UNK in real export)
    e[ids]=fq(emb[ids],ebits,g); m.bert.embeddings.word_embeddings.weight.data=e
    nbits+=len(ids)*768*ebits+(len(ids)*768//g)*32
    nbits+=(512+2)*768*16+768*768*16+768*2*16+2*768*16  # pos/type emb, pooler, classifier fp16
    p=preds(m)
    return dict(cfg=cfg, acc=float((p==y).mean()), agree=float((p==base).mean()), base_acc=float((base==y).mean()),
                bytes=nbits//8, ratio=nbits/8/BASE_BYTES, n_vocab=len(ids))

@app.local_entrypoint()
def main():
    layer_sets={"12":list(range(12))}
    cfgs=[dict(keep=k,ffn=f,bits=b,g=g,ebits=eb,ln=ln) for ln,k in layer_sets.items()
          for f in (1.0,) for b in (16,8,4,3) for g in (64,128) for eb in (16,8,4)]
    cfgs=[dict(c,vp=vp) for c in cfgs for vp in (0,1)]
    res=list(run.map(cfgs))
    json.dump(res,open("results3.json","w"))
    ok=[r for r in res if r["agree"]>=0.985]; ok.sort(key=lambda r:r["bytes"])
    for r in ok[:15]: print(r["ratio"],r["agree"],r["acc"],r["base_acc"],r["cfg"])
    print(len(res),"configs;", len(ok),"with agree>=.99; best ratio overall:", min(r["ratio"] for r in res))
