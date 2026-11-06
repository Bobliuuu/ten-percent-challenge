from ana import *
S={s:ld(s) for s in ("calib","dev","pool")}
for s,(rows,L,FO,y) in S.items():
    p=L[:,:5].argmax(1); print(s,"acc",(p==y).mean(),"rel",(p==y)[y!=4].mean(),"irr",(p==y)[y==4].mean(),"predNone",(p==4).mean())
rows,L,FO,y=S["calib"]
h,keep=build_head(rows,FO,set()); print(len(keep))
H=H_(h,dict(PR0))
for s in ("calib","pool","dev"):
    rows,L,FO,y=S[s]; U=H.proj(FO); G=H.gscore(U)
    for lam in (0,0.1,0.2,0.3,0.5,1.0):
        H.pr["lam0"]=lam; p=predict_all(H,P.Mem(),rows,L,U,G); print(s,lam,round((p==y).mean(),4),round((p==y)[y==4].mean(),3))
