from ana import *
rows,L,FO,y=ld("calib"); h,_=build_head(rows,FO,set()); pr=dict(PR0); pr["lam0"]=1.0; H=H_(h,pr)
dr,dL,dFO,dy=ld("dev"); C=Ctx(H,dr,dL,dFO,dy)
rng=np.random.default_rng(0); M=P.Mem(); C.ingest(M,list(rng.choice(2000,12,replace=False)))
v=C.pred(M); idx=rng.choice(2000,150,replace=False)
print(sum(int(C.row_scores(M,i).argmax())!=v[i] for i in idx), len(M.proto), memsize(M))
