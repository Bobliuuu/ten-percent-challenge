# Splits fixed by seed BEFORE any building. SST-2 validation(872) -> locked exam; part of train -> dev (for choosing).
import torch, json, numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
M="textattack/bert-base-uncased-SST-2"
ds=load_dataset("nyu-mll/glue","sst2")
rng=np.random.RandomState(0)
idx=rng.permutation(len(ds["train"]))
dev=idx[:2000].tolist(); feedback=idx[2000:4000].tolist()
json.dump({"dev":dev,"feedback":feedback},open("splits.json","w"))
tok=AutoTokenizer.from_pretrained(M); m=AutoModelForSequenceClassification.from_pretrained(M).eval()
print("params",sum(p.numel() for p in m.parameters()), m.config.id2label)
def acc(texts,labels):
    p=[]
    with torch.no_grad():
        for i in range(0,len(texts),64):
            b=tok(texts[i:i+64],padding=True,truncation=True,max_length=64,return_tensors="pt")
            p+=m(**b).logits.argmax(-1).tolist()
    return float(np.mean(np.array(p)==np.array(labels)))
d=ds["train"].select(dev)
print("DEV base acc",acc(d["sentence"],d["label"]))
