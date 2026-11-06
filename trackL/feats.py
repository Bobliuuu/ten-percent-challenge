import sys,json,pickle,numpy as np,torch,time
sys.path.insert(0,'/Users/jhome/ten-percent/trackD'); sys.path.insert(0,'/Users/jhome/ten-percent/trackG')
from lib import *
from dev2 import load_dev2
def texts_all():
    P,L=pickle.load(open('/Users/jhome/ten-percent/trackG/pool_logits.pkl','rb'))
    t2,y2=load_dev2(); th,yh=load_dev2(hard=True)
    ex=set(t2)|set(th)
    pool=[p[0] for p in P if p[0] not in ex]
    return pool,t2,th
@torch.no_grad()
def extract(m,texts,dev=None):
    dev=dev or dev_dev; m=m.to(dev); out={"logits":[],"pooled":[],"cls":[],"mean":[]}
    for b in batches(texts,64):
        b={k:v.to(dev) for k,v in b.items()}
        o=m(**b,output_hidden_states=True); h=o.hidden_states[-1]; am=b["attention_mask"].unsqueeze(-1).float()
        out["logits"].append(o.logits.float().cpu()); out["cls"].append(h[:,0].float().cpu())
        out["mean"].append(((h*am).sum(1)/am.sum(1)).float().cpu()); out["pooled"].append(m.bert.pooler(h).float().cpu())
    m.cpu(); return {k:torch.cat(v).numpy() for k,v in out.items()}
def all_feats(m):
    pool,t2,th=texts_all(); return {n:extract(m,t) for n,t in [("pool",pool),("d2",t2),("hard",th)]}
if __name__=="__main__":
    import gq
    cfg=json.loads(sys.argv[1]); name=cfg["name"]; t=time.time()
    if cfg.get("base"):
        m=load()
        F=all_feats(m); r={"name":name,"bytes_ent":437958644}
    else:
        if "bitsfor" in cfg:
            spec=cfg.pop("bitsfor"); hi=set(spec["names"]); hb=spec["bits"]; lb=cfg["bits"]
            cfg["bitsfor"]=lambda n,hi=hi,hb=hb,lb=lb:(hb if n in hi else lb)
        r=gq.run(cfg); m=r.pop("_m"); F=all_feats(m)
    pickle.dump((F,{k:v for k,v in r.items() if k!='bitsfor'}),open(f"F_{name}.pkl","wb")); print(json.dumps({k:v for k,v in r.items() if k!='bitsfor'}),time.time()-t)
