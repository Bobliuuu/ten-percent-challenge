import numpy as np
class ResidualMemory:
    """Gated residual linear memory on frozen features (no base weights touched).
    delta_logit = alpha * clip(w.x_c + b); trained only by mistake-driven online updates,
    with a small replay buffer of 'old' anchors (int8) to keep old behaviour. Stored fp16 w + int8 anchors."""
    def __init__(s,dim,mu,lr=0.1,margin=1.0,l2=1e-3,cap_anchor=300,epochs=30,seed=0):
        s.mu=mu.astype(np.float32); s.sd=1.0; s.w=np.zeros(dim,np.float32); s.b=0.0; s.lr,s.margin,s.l2,s.epochs=lr,margin,l2,epochs
        s.A=np.zeros((0,dim),np.int8); s.Ay=np.zeros(0,np.int8); s.cap=cap_anchor; s.rng=np.random.RandomState(seed); s.pend=[]
    def _x(s,E):
        E=np.atleast_2d(E).astype(np.float32)-s.mu; return 20*E/np.maximum(np.linalg.norm(E,axis=1,keepdims=True),1e-6)
    def score(s,E): return s._x(E)@s.w+s.b
    def predict(s,base_logits,embedding):
        z=s.score(embedding); bl=np.atleast_2d(base_logits).astype(np.float32)
        return bl+np.stack([-z,z],1)/2*1.0   # z is a logit-difference residual
    def add_anchor(s,embedding,label):
        x=np.clip(np.round(s._x(embedding)/20*127*4),-127,127).astype(np.int8)
        if len(s.Ay)<s.cap: s.A=np.vstack([s.A,x]); s.Ay=np.append(s.Ay,np.int8(label))
    def add(s,embedding,label,base_logits=None):
        s.pend.append((np.asarray(embedding,np.float32),int(label),None if base_logits is None else np.asarray(base_logits,np.float32)))
    def fit_pending(s):
        """consolidate the pending feedback batch: mistake-driven passes over (batch + anchors)"""
        if not s.pend: return
        Xn=s._x(np.stack([p[0] for p in s.pend])); yn=np.array([p[1] for p in s.pend]); bn=np.stack([p[2][1]-p[2][0] for p in s.pend])
        Xa=s.A.astype(np.float32)/(127*4)*20; ya=s.Ay.astype(int); ba=s.Abase if hasattr(s,'Abase') and len(s.Abase)==len(ya) else np.zeros(len(ya))
        X=np.vstack([Xn,Xa]); y=np.concatenate([yn,ya]); bb=np.concatenate([bn,ba]); sg=2*y-1
        for ep in range(s.epochs):
            for i in s.rng.permutation(len(y)):
                z=bb[i]+X[i]@s.w+s.b
                if sg[i]*z<s.margin*(8 if i<len(yn) else 4):   # hinge on combined (base+residual) margin
                    s.w+=s.lr*sg[i]*X[i]; s.b+=s.lr*sg[i]*0.1
                s.w*=1-s.l2
        s.w=s.w.astype(np.float16).astype(np.float32); s.pend=[]
    def bytes(s): return int(s.w.size*2+2+s.A.nbytes+s.Ay.nbytes+s.mu.size*2)
