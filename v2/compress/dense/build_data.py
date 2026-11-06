"""Build disjoint calib / ID-eval / OOD-eval / teacher-forced sets from public datasets. Deterministic (seed 0)."""
import json, random, re
from datasets import load_dataset
R=random.Random(0)
def take(it,n,skip=0,flt=lambda r:True):
    out=[];i=0
    for r in it:
        if not flt(r): continue
        i+=1
        if i<=skip: continue
        out.append(r)
        if len(out)>=n: break
    return out
dolly=list(load_dataset("databricks/databricks-dolly-15k",split="train")); R.shuffle(dolly)
def dl(cats,n,off):
    c=[r for r in dolly if r["category"] in cats]; return c[off:off+n]
def dprompt(r): return (r["instruction"]+("\n\n"+r["context"][:1500] if r["context"] else ""))
pools={}
# per kind: pool of (prompt, ref_response); 3 disjoint slices: eval(100) tf(30) calib(rest)
pools["chat_qa"]=[(dprompt(r),r["response"]) for r in dl({"open_qa","general_qa","brainstorming"},1000,0)]
pools["sumcls"]=[(dprompt(r),r["response"]) for r in dl({"summarization","classification","closed_qa"},1000,0) if len(dprompt(r))<2500]
gl=take(iter(load_dataset("glaiveai/glaive-function-calling-v2",split="train",streaming=True)),1500)
fc=[]
for r in gl:
    m=re.match(r"USER: (.*?)\n\n\nASSISTANT: (.*?)(<\|endoftext\|>|$)",r["chat"],re.S)
    if m and "functioncall" in r["chat"][:3000] and len(r["system"])<2500:
        fc.append(("[System]\n"+r["system"].replace("SYSTEM: ","")+"\n\n[User]\n"+m.group(1)+"\n\nRespond with the function call as JSON only if a function applies.",m.group(2).strip()))
pools["json_fc"]=fc
pii=take(iter(load_dataset("ai4privacy/pii-masking-200k",split="train",streaming=True)),1500,flt=lambda r:r["language"]=="en")
pools["pii"]=[("Rewrite the following text, replacing every piece of personal information with a bracketed placeholder such as [NAME] or [PHONE]:\n\n"+r["source_text"],r["target_text"]) for r in pii]
ev_id=[];tf=[];cal=[]
for k,p in pools.items():
    R.shuffle(p); print(k,len(p))
    ev_id+=[dict(kind=k,prompt=a) for a,b in p[:100]]
    tf+=[dict(kind=k,prompt=a,ref=b) for a,b in p[100:130]]
    cal+=[dict(prompt=a,ref=b) for a,b in p[200:200+190]]
# OOD
ood=[]
g=list(load_dataset("openai/gsm8k","main",split="test")); R.shuffle(g)
ood+=[dict(kind="math",prompt=r["question"]) for r in g[:50]]
h=list(load_dataset("openai/openai_humaneval",split="test")); R.shuffle(h)
ood+=[dict(kind="code",prompt="Complete the following Python function:\n\n"+r["prompt"]) for r in h[:50]]
c=take(iter(load_dataset("abisee/cnn_dailymail","3.0.0",split="test",streaming=True)),50)
ood+=[dict(kind="news_sum",prompt="Summarize the following article in two sentences:\n\n"+r["article"][:2500]) for r in c]
a=take(iter(load_dataset("CohereLabs/aya_dataset",split="test",streaming=True)),3000,flt=lambda r:r["language_code"] not in("eng",))
R.shuffle(a)
ood+=[dict(kind="multiling",prompt=r["inputs"][:1500]) for r in a[:50]]
json.dump(dict(eval_id=ev_id,eval_ood=ood,tf=tf,calib=cal),open("data/dense_data.json","w"))
print(len(ev_id),len(ood),len(tf),len(cal))
