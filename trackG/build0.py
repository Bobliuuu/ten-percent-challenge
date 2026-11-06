import re, numpy as np, torch, pickle
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
rng=np.random.RandomState(1)
out={}
im=load_dataset("stanfordnlp/imdb",split="train")
ix=rng.permutation(len(im))[:3000]
S=[]
for i in ix:
    t=re.sub(r"<br\s*/?>"," ",im[int(i)]["text"])
    ss=[s.strip() for s in re.split(r"(?<=[.!?])\s+",t) if 6<=len(s.split())<=40]
    if ss: 
        # keep sentences w/ strong review label only: sentence label = review label (noisy)
        for s in rng.choice(ss,min(3,len(ss)),replace=False): S.append((str(s),im[int(i)]["label"],"imdb"))
out["imdb"]=S
for name,hf,col in [("yelp","fancyzeng/yelp_polarity" ,None)]:
    pass
y=load_dataset("fancyzeng/yelp_polarity",split="train") if False else None
pickle.dump(out,open("raw_imdb.pkl","wb")); print(len(S))
