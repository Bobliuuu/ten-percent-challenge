"""Build feedback data for L7 from public MASSIVE (cooking scenario = NEW skill 'Cooking'). Hash buckets as EASY (easy-split-v1): [0,40) feedback pool, [40,70) teach-eval pool (unseen), [70,100) never used."""
import json, hashlib, re
from datasets import load_dataset
H=lambda s:int(hashlib.sha256(s.encode()).hexdigest(),16)%100
norm=lambda t:re.sub(r"[^a-z0-9 ]","", " ".join(str(t).lower().split()))
ms=load_dataset("mteb/amazon_massive_scenario",name="en",split="train")
rows=sorted({norm(x["text"]):x["text"] for x in ms if x["label_text"]=="cooking"}.items())
P=open("../../task_search/EASY/PLAN.json").read(); P=json.loads(P)["SKILL7"]["prompt"]
fb=[];te=[]
for n,t in rows:
    b=H(f"easy-split-v1|COOK|{n}")
    (fb if b<40 else te if b<70 else []).append(t)
print(len(rows),len(fb),len(te))
import random; random.Random(1).shuffle(fb); random.Random(2).shuffle(te)
json.dump({"fb25":fb[:25],"fb_extra":fb[25:125],"teach_unseen":te[:100],"prompt":P},open("data/cook.json","w"),indent=1)
