"""v2/alt: build MDC-6 (multi-domain text classification) splits. Run: modal run --detach build_data.py
Tasks: ag_news(topic,4) dbpedia_14(entity type,14) yelp_polarity(2) amazon_polarity(2) emotion(6) tweet_eval/offensive(2 toxicity)
Split rule (hash, deterministic): h=int(sha256(task|text|label)[:8],16); over each task's TEST split h%7==0 -> LOCKED, ==1 -> DEV, else POOL.
calib = TRAIN split samples (disjoint from test). Caps per task: locked<=400 dev<=400 pool<=800 calib<=600 (items taken in ascending hash order).
Output: Volume alt-data:/mdc6/{locked,dev,pool,calib}.jsonl + RESULT_JSON with counts and sha256.
"""
import modal, json
app = modal.App("tenpct-alt-data")
img = modal.Image.debian_slim().pip_install("datasets<4","pyarrow")
vol = modal.Volume.from_name("alt-data", create_if_missing=True)
TASKS = {
 "ag_news":  dict(hf=("fancyzhx/ag_news",None), tr="train", te="test", text="text", lab="label", names=["World","Sports","Business","Science/Technology"], inst="Classify the topic of the news article."),
 "dbpedia":  dict(hf=("fancyzhx/dbpedia_14",None), tr="train", te="test", text="content", lab="label",
    names=["Company","Educational institution","Artist","Athlete","Office holder","Mean of transportation","Building","Natural place","Village","Animal","Plant","Album","Film","Written work"], inst="Classify the type of entity the Wikipedia abstract describes."),
 "yelp":     dict(hf=("fancyzhx/yelp_polarity",None), tr="train", te="test", text="text", lab="label", names=["Negative","Positive"], inst="Classify the sentiment of the restaurant/business review."),
 "amazon":   dict(hf=("fancyzhx/amazon_polarity",None), tr="train", te="test", text=("title","content"), lab="label", names=["Negative","Positive"], inst="Classify the sentiment of the product review."),
 "emotion":  dict(hf=("dair-ai/emotion","split"), tr="train", te="test", text="text", lab="label", names=["Sadness","Joy","Love","Anger","Fear","Surprise"], inst="Classify the main emotion expressed in the message."),
 "offensive":dict(hf=("cardiffnlp/tweet_eval","offensive"), tr="train", te="test", text="text", lab="label", names=["Not offensive","Offensive"], inst="Moderation: decide whether the social media post is offensive or toxic."),
}
CAP=dict(locked=400,dev=400,pool=800,calib=600)
@app.function(image=img, timeout=3600, volumes={"/d":vol})
def build():
    import hashlib, os
    from datasets import load_dataset
    out={k:[] for k in CAP}; info={}
    for tn,t in TASKS.items():
        try:
            ds=load_dataset(t["hf"][0],t["hf"][1]) if t["hf"][1] else load_dataset(t["hf"][0])
        except Exception as e:
            print("FAIL",tn,repr(e)); info[tn]="FAIL "+repr(e); continue
        def rows(split):
            r=[]
            for ex in ds[split]:
                tx=ex[t["text"]] if isinstance(t["text"],str) else ". ".join(ex[c] for c in t["text"])
                tx=" ".join(tx.split())[:1200]
                r.append(dict(task=tn,text=tx,answer=int(ex[t["lab"]])))
            return r
        te=rows(t["te"]); 
        for r in te: r["h"]=int(hashlib.sha256(f"{tn}|{r['text']}|{r['answer']}".encode()).hexdigest()[:8],16)
        te.sort(key=lambda r:r["h"])
        b={"locked":[r for r in te if r["h"]%7==0],"dev":[r for r in te if r["h"]%7==1],"pool":[r for r in te if r["h"]%7>1]}
        tr=ds[t["tr"]]; import random
        idx=random.Random(0).sample(range(len(tr)),CAP["calib"]); trr=[]
        full=rows(t["tr"]) if len(tr)<50000 else None
        if full is None:
            full=[]
            for i in idx:
                ex=tr[i]; tx=ex[t["text"]] if isinstance(t["text"],str) else ". ".join(ex[c] for c in t["text"])
                full.append(dict(task=tn,text=" ".join(tx.split())[:1200],answer=int(ex[t["lab"]])))
            b["calib"]=full
        else:
            b["calib"]=[full[i] for i in idx]
        for k in CAP:
            for r in b[k][:CAP[k]]:
                out[k].append(dict(id=f"{tn}-{k}-{len(out[k])}",task=tn,text=r["text"],answer=r["answer"],names=t["names"],inst=t["inst"]))
        info[tn]={k:min(len(b[k]),CAP[k]) for k in CAP}
    os.makedirs("/d/mdc6",exist_ok=True); res={"info":info,"sha":{}}
    for k,v in out.items():
        p=f"/d/mdc6/{k}.jsonl"
        with open(p,"w") as f:
            for r in v: f.write(json.dumps(r,ensure_ascii=False)+"\n")
        res["sha"][k]=hashlib.sha256(open(p,"rb").read()).hexdigest(); res.setdefault("n",{})[k]=len(v)
    vol.commit(); print("RESULT_JSON "+json.dumps(res)); return res
@app.local_entrypoint()
def main(): build.remote()
