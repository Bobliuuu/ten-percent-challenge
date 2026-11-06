import sys,pickle,hashlib,numpy as np,torch
sys.path.insert(0,'/Users/jhome/ten-percent/trackG'); sys.path.insert(0,'/Users/jhome/ten-percent/trackH') if False else None
from dev2 import load_dev2
from feats import texts_all
torch.manual_seed(0)
FB,_=pickle.load(open("F_base.pkl","rb")); FA,_=pickle.load(open("F_A.pkl","rb"))
pool,t2,th=texts_all(); _,y2=load_dev2()
h=np.array([int(hashlib.md5(t.encode()).hexdigest(),16)%2 for t in t2]); ev=h==1; cal2=h==0
y2=np.array(y2)
d=lambda F,k:(F[k]["logits"][:,1]-F[k]["logits"][:,0])
bd={k:d(FB,k) for k in FB}; qd={k:d(FA,k) for k in FA}
def rep(name,zp,zh,extra=""):
    # zp: corrected diff on d2 (all 3000); zh: on hard
    ag=((zp>0)==(bd["d2"]>0))[ev].mean(); ah=((zh>0)==(bd["hard"]>0)).mean(); acc=((zp>0)==y2)[ev].mean()
    print(f"{name:42s} agree_d2ev {ag:.4f} agree_hard {ah:.4f} acc_d2ev {acc:.4f} {extra}",flush=True)
    return ag,ah
rep("quantized (no corr)",qd["d2"],qd["hard"],f"base acc {((bd['d2']>0)==y2)[ev].mean():.4f}")
# pool stats
print("pool agree",((qd["pool"]>0)==(bd["pool"]>0)).mean(), len(pool))
def LR(X,y,w=None,l2=1e-3,ep=300,lr=0.05):
    X=torch.tensor(X,dtype=torch.float32); y=torch.tensor(y,dtype=torch.float32)
    W=torch.zeros(X.shape[1],requires_grad=True); b=torch.zeros(1,requires_grad=True)
    o=torch.optim.LBFGS([W,b],lr=1,max_iter=ep,line_search_fn="strong_wolfe")
    def cl():
        o.zero_grad(); l=torch.nn.functional.binary_cross_entropy_with_logits(X@W+b,y)+l2*(W**2).sum(); l.backward(); return l
    o.step(cl); return W.detach().numpy(),b.item()
yb=(bd["pool"]>0).astype(np.float32)
# (a) bias/temperature
W,b=LR(qd["pool"][:,None],yb); print("(a) a,b",W,b)
f=lambda k:W[0]*qd[k]+b
rep("(a) scale+bias (8 bytes)",f("d2"),f("hard"),"bytes 8")
# (b) linear on features
for feat in ["pooled","cls","mean"]:
    for l2 in [1e-4,1e-3,1e-2]:
        X=lambda k:np.concatenate([qd[k][:,None],FA[k][feat]],1)
        W,b=LR(X("pool"),yb,l2=l2)
        rep(f"(b) LR[qdiff,{feat}] l2={l2}",X("d2")@W+b,X("hard")@W+b,"bytes 1.5K fp16")
# soft target regression (ridge) on pooled
from numpy.linalg import solve
for feat in ["pooled","cls"]:
  for lam in [1,10,100]:
    X=lambda k:np.concatenate([qd[k][:,None],FA[k][feat],np.ones((len(qd[k]),1))],1)
    Xp=X("pool"); w=solve(Xp.T@Xp+lam*np.eye(Xp.shape[1]),Xp.T@bd["pool"])
    rep(f"(b') ridge->base diff [{feat}] lam={lam}",X("d2")@w,X("hard")@w)
pickle.dump(None,open("/dev/null","wb"))
