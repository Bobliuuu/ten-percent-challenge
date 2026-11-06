# caches pooled embeddings + base logits for dev/feedback (train-split only; never touches validation/test) + synthetic slang set
import torch, json, numpy as np, random
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
M="textattack/bert-base-uncased-SST-2"
tok=AutoTokenizer.from_pretrained(M); m=AutoModelForSequenceClassification.from_pretrained(M).eval()
dev_=torch.device("mps" if torch.backends.mps.is_available() else "cpu"); m.to(dev_)
def run(texts):
    E=[];L=[]
    with torch.no_grad():
        for i in range(0,len(texts),64):
            b=tok(texts[i:i+64],padding=True,truncation=True,max_length=64,return_tensors="pt").to(dev_)
            o=m.bert(**b,output_hidden_states=True); h=o.pooler_output; L.append(m.classifier(h).cpu()); mk=b['attention_mask'].unsqueeze(-1).float(); hs=o.hidden_states[-1]; mean=(hs*mk).sum(1)/mk.sum(1); E.append(torch.cat([h,hs[:,0],mean],1).cpu())
    return torch.cat(E).numpy(),torch.cat(L).numpy()
S=json.load(open("../splits.json")); tr=load_dataset("nyu-mll/glue","sst2")["train"]
out={}
for k in ["dev","feedback"]:
    d=tr.select(S[k]); e,l=run(d["sentence"]); out[k+"_e"]=e;out[k+"_l"]=l;out[k+"_y"]=np.array(d["label"]); out[k+"_t"]=np.array(d["sentence"])
# synthetic new-concept: invented/new slang terms, templated
random.seed(0)
POS=["bussin","slaps","goated","zorbly","glimmerful","fleek","rizzy","kerblam","snazzle","vibeworthy","poggers","crispy"]
NEG=["mid","cringe","blenky","yeeted","flopsy","grotty","dreckish","snoozefest","wonky-tier","meh-core","bricked","sus"]
SUB=["the movie","this film","the acting","the soundtrack","the plot","the ending","the director's cut","the script","the sequel","the cinematography","the performance","the dialogue","this documentary","the cast"]
TPL=["{s} was {w}.","honestly {s} is totally {w}","{s}? absolutely {w}","i found {s} {w} from start to finish","{s} felt kinda {w} to me","wow, {s} is so {w}"]
def gen(words,y,n):
    return [(random.choice(TPL).format(s=random.choice(SUB),w=w),y) for w in words for _ in range(n)]
syn=[]
for w in POS: syn+= [(t,1,w) for t,_ in gen([w],1,40)]
for w in NEG: syn+= [(t,0,w) for t,_ in gen([w],0,40)]
txt=[s[0] for s in syn]; e,l=run(txt)
out["syn_t"]=np.array(txt);out["syn_e"]=e;out["syn_l"]=l;out["syn_y"]=np.array([s[1] for s in syn]);out["syn_w"]=np.array([s[2] for s in syn])
np.savez("cache.npz",**out)
for k in ["dev","feedback","syn"]: print(k,(out[k+"_l"].argmax(1)==out[k+"_y"]).mean())
