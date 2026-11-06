import numpy as np
class Memory:
    """Frozen-base feedback memory: int8 kNN over L2-normalised pooled embeddings with
    similarity-gated additive logit bias. No weights are trained.
    add(embedding,label) ; predict(base_logits, embedding)->logits ; bytes()"""
    def __init__(s, dim=768, cap=1000, k=5, tau=0.90, alpha=8.0, T=0.02, dedup=0.995, mu=None):
        s.mu=None if mu is None else np.asarray(mu,np.float32)
        s.dim,s.cap,s.k,s.tau,s.alpha,s.T,s.dedup=dim,cap,k,tau,alpha,T,dedup
        s.X=np.zeros((0,dim),np.int8); s.y=np.zeros(0,np.int8); s.hits=np.zeros(0,np.int32); s.age=np.zeros(0,np.int32); s.t=0
    def _q(s,e):
        e=np.asarray(e,np.float32); e=e if s.mu is None else e-s.mu; e=e/np.linalg.norm(e,axis=-1,keepdims=True)
        return np.clip(np.round(e*127*4),-127,127).astype(np.int8)  # fixed scale: unit-vector comps ~<0.5
    def _sims(s,E):
        A=E.astype(np.float32); B=s.X.astype(np.float32)
        return (A@B.T)/np.maximum(np.linalg.norm(A,axis=1,keepdims=True)*np.linalg.norm(B,axis=1)[None],1e-6)
    def add(s,embedding,label):
        E=s._q(embedding[None]); s.t+=1
        if len(s.y):
            sm=s._sims(E)[0]; j=sm.argmax()
            if sm[j]>=s.dedup:
                if s.y[j]==label: s.hits[j]+=1; return False   # duplicate
                s.y[j]=label; return True                      # relabel
        if len(s.y)>=s.cap:  # evict least-used, oldest
            v=s.hits*1000-(s.t-s.age)*0+s.age; i=int(np.argmin(s.hits*100000+s.age)); keep=np.arange(len(s.y))!=i
            s.X,s.y,s.hits,s.age=s.X[keep],s.y[keep],s.hits[keep],s.age[keep]
        s.X=np.vstack([s.X,E]); s.y=np.append(s.y,np.int8(label)); s.hits=np.append(s.hits,0).astype(np.int32); s.age=np.append(s.age,s.t).astype(np.int32)
        return True
    def vote(s,embedding):
        E=s._q(np.atleast_2d(embedding)); out=np.zeros(len(E),np.float32)
        if not len(s.y): return out
        sm=s._sims(E); k=min(s.k,sm.shape[1]); idx=np.argpartition(-sm,k-1,axis=1)[:,:k]
        for r in range(len(E)):
            ss=sm[r,idx[r]]; m=ss>=s.tau
            if m.any():
                w=np.exp((ss[m]-ss[m].max())/s.T)*ss[m]; sg=2.0*s.y[idx[r][m]]-1
                out[r]=(w*sg).sum()/w.sum()*(1 if True else 0)   # in [-1,1]
        return out
    def predict(s,base_logits,embedding):
        v=s.vote(embedding); base_logits=np.atleast_2d(base_logits).astype(np.float32)
        return base_logits+s.alpha*np.stack([-v,v],1)
    def bytes(s): return int(s.X.nbytes+s.y.nbytes)  # payload: int8 vectors + labels (hits/age are bookkeeping, 1B each if kept)
