#!/usr/bin/env python
"""Build frozen v2 splits for Qwen2.5-14B-Instruct / MMLU. Refuses to overwrite LOCK.json.
Rule (same as trackR): h=int(sha256(subject|question|choice0|..|choice3)[:8],16);
 test items: h%7==0 -> LOCKED exam, h%7==1 -> DEV, else -> POOL (buildable, e.g. feedback/calib extras).
CALIB = MMLU validation + MMLU dev(few-shot) + 2000 seeded auxiliary_train items (disjoint from test).
Run: python build_splits.py   (needs `datasets`; light, CPU)"""
import hashlib, json, os, datetime, numpy as np
from datasets import load_dataset
HERE=os.path.dirname(os.path.abspath(__file__))
assert not os.path.exists(f"{HERE}/LOCK.json"), "LOCK.json exists; never overwrite"
def h(r): return int(hashlib.sha256(("|".join([r["subject"],r["question"]]+list(r["choices"]))).encode()).hexdigest()[:8],16)
ds=load_dataset("cais/mmlu","all"); T=ds["test"]
def row(i,r,src): return dict(id=f"{src}{i}",subject=r["subject"],question=r["question"],choices=list(r["choices"]),answer=int(r["answer"]))
S={"locked":[], "dev":[], "pool":[], "calib":[]}
for i,r in enumerate(T):
    S[{0:"locked",1:"dev"}.get(h(r)%7,"pool")].append(row(i,r,"t"))
for i,r in enumerate(ds["validation"]): S["calib"].append(row(i,r,"v"))
for i,r in enumerate(ds["dev"]): S["calib"].append(row(i,r,"d"))
A=ds["auxiliary_train"]; 
for i in np.random.RandomState(0).permutation(len(A))[:2000]: S["calib"].append(row(int(i),A[int(i)],"a"))
def sha(rows): return hashlib.sha256("".join(f"{r['id']}\t{r['answer']}\t{r['question']}\t{'|'.join(r['choices'])}\n" for r in rows).encode()).hexdigest()
lock=dict(rule=__doc__.split("\n")[1:3], hash_def="sha256 over rows in file order of f'{id}\\t{answer}\\t{question}\\t{choices joined by |}\\n'",
          utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), dataset="cais/mmlu all", splits={})
# overlap sanity (question text)
q=lambda k:{r["question"] for r in S[k]}
lock["overlap_locked_dev"]=len(q("locked")&q("dev")); lock["overlap_locked_calib"]=len(q("locked")&q("calib")); lock["overlap_locked_pool"]=len(q("locked")&q("pool"))
for k,rows in S.items():
    p=f"{HERE}/{k}.jsonl"
    with open(p,"w") as f:
        for r in rows: f.write(json.dumps(r,ensure_ascii=False)+"\n")
    lock["splits"][k]=dict(n=len(rows),content_sha256=sha(rows),file_sha256=hashlib.sha256(open(p,"rb").read()).hexdigest())
    os.chmod(p,0o444)
json.dump(lock,open(f"{HERE}/LOCK.json","w"),indent=1); os.chmod(f"{HERE}/LOCK.json",0o444)
print(json.dumps(lock,indent=1))
