# Hash-fixed MMLU split. locked = test items with h%7==0 ; sweep-dev = h%7==1 ; calib text = MMLU validation+dev (+ wikitext).
import hashlib, json
from datasets import load_dataset
def h(r): return int(hashlib.sha256(("|".join([r["subject"],r["question"]]+list(r["choices"]))).encode()).hexdigest()[:8],16)
if __name__=="__main__":
    t=load_dataset("cais/mmlu","all")["test"]
    L=[i for i,r in enumerate(t) if h(r)%7==0]; D=[i for i,r in enumerate(t) if h(r)%7==1]
    sha=lambda ids: hashlib.sha256(("\n".join(f"{i}\t{t[i]['answer']}\t{t[i]['question']}" for i in ids)).encode()).hexdigest()
    import datetime
    json.dump(dict(rule="sha256(subject|question|choices)[:8] mod 7: 0=locked,1=dev",n_test=len(t),n_locked=len(L),n_dev=len(D),locked_sha256=sha(L),dev_sha256=sha(D),utc=datetime.datetime.utcnow().isoformat()),open("LOCK.json","w"),indent=1)
    print(len(t),len(L),len(D))
