import pickle, json, numpy as np
P,L=pickle.load(open("pool_logits.pkl","rb"))
y=np.array([p[1] for p in P]); src=np.array([p[2] for p in P])
pred=L.argmax(1); mar=np.abs(L[:,1]-L[:,0]); ok=pred==y
rng=np.random.RandomState(123)
quota={"imdb":1200,"amazon_title":1000,"tweet":800}
sel=[]
for s,q in quota.items():
    nw=int(round(q*0.07)); nc=q-nw
    for correct,n in [(True,nc),(False,nw)]:
        for lab in (0,1):
            c=np.where((src==s)&(ok==correct)&(y==lab))[0]
            sel+=rng.choice(c,n//2,replace=False).tolist()
sel=np.array(sel); rng.shuffle(sel)
rest=np.setdiff1d(np.arange(len(P)),sel)
hard=rest[np.argsort(mar[rest])[:1500]]
def dump(ix,f):
    json.dump([{"text":P[i][0],"label":int(P[i][1]),"source":P[i][2],"base_margin":round(float(mar[i]),3),"base_pred":int(pred[i])} for i in ix],open(f,"w"))
dump(sel,"dev2.json"); dump(hard,"dev2_hard.json")
for n,ix in [("dev2",sel),("hard",hard)]:
    print(n,len(ix),"acc",ok[ix].mean(),"pos",y[ix].mean(),"margin pcts",np.percentile(mar[ix],[10,25,50,75,90]).round(2),"<1:",(mar[ix]<1).mean(),"<3",(mar[ix]<3).mean())
