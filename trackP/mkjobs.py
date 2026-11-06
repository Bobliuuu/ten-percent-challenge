import json,sys
from allocate import knap
def cfg_from(ch,**kw):
    enc={k:list(v) for k,v in ch.items() if k.startswith("L")}
    emb=[list(ch[f"emb{j}"]) for j in range(5)]
    return dict(enc=enc,emb=emb,pool=list(ch["pool"]),**kw)
if __name__=="__main__":
    s=json.load(open("sens.json")); jobs=[]
    for bud in [42.6e6,43.0e6,43.3e6]:
        ch,tot,kl=knap(s,bud); jobs.append(dict(name=f"k{int(bud/1e5)}",cfg=cfg_from(ch,calib="dev"),keep=False)); print(bud,tot,kl)
    # trackA reference config
    SH=dict(q="attention.self.query",k="attention.self.key",v="attention.self.value",a="attention.output.dense",i="intermediate.dense",o="output.dense")
    B=dict(q=(2,128),k=(2,128),v=(3,128),a=(3,128),o=(3,256)); enc={}
    for l in range(12):
        for c,n in SH.items(): enc[f"L{l}.{n}"]=list((4 if l<6 else 3,128) if c=="i" else B[c])
    jobs.append(dict(name="trackA",cfg=dict(enc=enc,emb=[[2,128]]+[[3,128]]*4,pool=[6,64],calib="dev"),keep=False))
    json.dump(jobs,open("jobs1.json","w"))
