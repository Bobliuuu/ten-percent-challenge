import json,sys,torch,collections,time
sys.path.insert(0,".")
import embq
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
M="textattack/bert-base-uncased-SST-2"
torch.set_num_threads(10)
dev="cpu"
tok=AutoTokenizer.from_pretrained(M); m=AutoModelForSequenceClassification.from_pretrained(M).to(dev).eval()
D=lambda f:[(x["text"],x["label"]) for x in json.load(open(f))]
d2=D("dev2.json")[512:]; dh=D("dev2_hard.json")
def P(data):
    o=[]
    with torch.no_grad():
        for i in range(0,len(data),64):
            b=tok([t for t,_ in data[i:i+64]],padding=True,truncation=True,max_length=128,return_tensors="pt").to(dev)
            o.append(m(**b).logits.argmax(-1).cpu())
    return torch.cat(o)
E0=m.get_input_embeddings().weight.data.clone().cpu(); V,H=E0.shape
b2=P(d2); bh=P(dh)
tr=list(load_dataset("nyu-mll/glue","sst2")["train"]["sentence"][:])
cnt=collections.Counter()
for e in tok(tr,truncation=True,max_length=128)["input_ids"]: cnt.update(e)
freq=torch.zeros(V)
for k,v in cnt.items(): freq[k]=v
for s in tok.all_special_ids: freq[s]=1e9
print("seen",int((freq>0).sum()),flush=True)
res=[]
for name,kw in embq.SCHEMES:
    t=time.time(); E,nb=embq.quantize_embedding(E0,freq,**kw)
    m.get_input_embeddings().weight.data=E.to(dev)
    r=dict(name=name,emb_MB=nb/1e6,agree=float((P(d2)==b2).float().mean()),agree_hard=float((P(dh)==bh).float().mean()))
    print(r,round(time.time()-t),flush=True); res.append(r)
    json.dump(res,open("emb_bert.json","w"),indent=1)
