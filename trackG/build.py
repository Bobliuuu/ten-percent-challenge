import re, json, pickle, numpy as np, torch
from datasets import load_dataset
rng=np.random.RandomState(7)
pool=[]
for t,l,s in pickle.load(open("raw_imdb.pkl","rb"))["imdb"]: pool.append((t,l,"imdb"))
am=load_dataset("mteb/amazon_polarity",split="train[:60000]")
for i in rng.permutation(len(am))[:6000]:
    r=am[int(i)]; t=r["text"].split("\n\n")[0].strip()
    if 4<=len(t.split())<=30: pool.append((t,r["label"],"amazon_title"))
te=load_dataset("cardiffnlp/tweet_eval","sentiment",split="train")
c=0
for i in rng.permutation(len(te)):
    r=te[int(i)]
    if r["label"] in (0,2):
        t=re.sub(r"@user","",r["text"]).strip()
        if 6<=len(t.split())<=40: pool.append((t,1 if r["label"]==2 else 0,"tweet")); c+=1
    if c>=4000: break
seen=set(); P=[]
for t,l,s in pool:
    k=re.sub(r"\W+"," ",t.lower()).strip()
    if k in seen: continue
    seen.add(k); P.append((t,l,s))
print(len(P))
from transformers import AutoTokenizer, AutoModelForSequenceClassification
M="textattack/bert-base-uncased-SST-2"
tok=AutoTokenizer.from_pretrained(M); m=AutoModelForSequenceClassification.from_pretrained(M).eval().to("mps")
L=[]
with torch.no_grad():
    for i in range(0,len(P),128):
        b=tok([p[0] for p in P[i:i+128]],padding=True,truncation=True,max_length=64,return_tensors="pt").to("mps")
        L+=m(**b).logits.float().cpu().tolist()
L=np.array(L); pickle.dump((P,L),open("pool_logits.pkl","wb"))
y=np.array([p[1] for p in P]); src=np.array([p[2] for p in P])
pred=L.argmax(1); mar=np.abs(L[:,1]-L[:,0])
for s in ["imdb","amazon_title","tweet"]:
    k=src==s; print(s,k.sum(),"acc",(pred[k]==y[k]).mean(),"pos%",y[k].mean(),"medmargin",np.median(mar[k]),"frac margin<2",(mar[k]<2).mean())
