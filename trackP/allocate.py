import json,sys,numpy as np
def knap(sens,budget,unit=2000,scale_kl=None):
    """sens: key 'item|bits|g' -> dict(kl,size). returns choice dict item->(bits,g), total size, total kl"""
    items={}
    for k,v in sens.items():
        it,b,g=k.split("|"); items.setdefault(it,[]).append((int(b),int(g),v["size"],max(v["kl"],0)))
    # prune dominated options
    names=sorted(items); B=int(budget//unit); INF=1e18
    dp=np.full(B+1,INF); dp[0]=0; choice=[]
    for it in names:
        opts=[]
        for o in sorted(items[it],key=lambda t:t[2]):
            if not opts or o[3]<opts[-1][3]: opts.append(o)
        nd=np.full(B+1,INF); arg=np.full(B+1,-1,dtype=np.int32)
        for j,(b,g,sz,kl) in enumerate(opts):
            c=int(np.ceil(sz/unit))
            if c>B: continue
            cand=np.full(B+1,INF); cand[c:]=dp[:B+1-c]+kl
            m=cand<nd; nd=np.where(m,cand,nd); arg=np.where(m,j,arg)
        choice.append((opts,arg)); dp=nd
    # dp[w]=min kl using exactly w units; take best over w<=B
    w=int(np.argmin(dp)); best=dp[w]; out={}; tot=0
    for it,(opts,arg) in zip(reversed(names),reversed(choice)):
        j=int(arg[w]); b,g,sz,kl=opts[j]; out[it]=(b,g); tot+=sz; w-=int(np.ceil(sz/unit))
    return out,tot,best
if __name__=="__main__":
    s=json.load(open(sys.argv[1])); bud=float(sys.argv[2])
    ch,tot,kl=knap(s,bud); print(tot,kl); 
    for k in sorted(ch): print(k,ch[k])
