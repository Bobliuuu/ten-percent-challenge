import numpy as np
class LexMemory:
    """Sparse lexical residual memory over frozen-base wordpiece ids (no base weights touched).
    delta (logit-diff) = sum_{t in tokens} w[t] / sqrt(n)  ; w stored sparse int8 (id int16 + w int8 = 3B/entry)
    + replay anchors (token-id lists, int16) of old data to keep old behaviour. Consolidation = mistake-driven
    hinge perceptron on (pending feedback + anchors) wrt combined base+memory margin."""
    def __init__(s,lr=0.5,margin=6.0,epochs=30,l2=0.0,scale=8.0,seed=0,max_df=10**9):
        s.w={}; s.lr,s.margin,s.epochs,s.l2,s.scale=lr,margin,epochs,l2,scale; s.rng=np.random.RandomState(seed)
        s.anc=[];s.pend=[];s.max_df=max_df;s.df={}
    def _d(s,toks):
        t=set(toks); 
        return sum(s.w.get(i,0.0) for i in t)/np.sqrt(max(len(t),1))
    def delta(s,toks_list): return np.array([s._d(t) for t in toks_list])
    def predict(s,base_logits,toks_list):
        z=s.delta(toks_list); bl=np.atleast_2d(base_logits).astype(np.float32); return bl+np.stack([-z,z],1)/2
    def add_anchor(s,toks,label,base_diff):
        for i in set(toks): s.df[i]=s.df.get(i,0)+1
        s.anc.append((list(set(toks)),int(label),float(base_diff)))
    def add(s,toks,label,base_diff): s.pend.append((list(set(toks)),int(label),float(base_diff)))
    def fit_pending(s):
        items=[(p,3.0) for p in s.pend]+[(a,1.0) for a in s.anc]
        for ep in range(s.epochs):
            for j in s.rng.permutation(len(items)):
                (t,y,b),wt=items[j]; sg=2*y-1; n=np.sqrt(len(t))
                if sg*(b+s._d(t))<s.margin:
                    for i in t:
                        if s.df.get(i,0)<=s.max_df: s.w[i]=s.w.get(i,0.0)+s.lr*sg*wt/n
            if s.l2:
                for i in s.w: s.w[i]*=1-s.l2
        s.w={i:float(np.clip(np.round(v*8),-127,127))/8 for i,v in s.w.items() if abs(v)>=0.125}  # int8 quantise (1/8 step), prune
        s.pend=[]
    def bytes(s): return 3*len(s.w)+sum(2*len(a[0])+1 for a in s.anc)

class Memory(LexMemory):
    """Clean API (recommended memory). 'embedding' = wordpiece-id list of the sentence (frozen-base tokenizer).
      m.add(tokens,label,base_logits)   queue feedback (base_logits needed for margin)
      m.add_anchor(tokens,label,base_logits)  old-behaviour replay item
      m.fit_pending()                   consolidate (mistake-driven, no base weight training)
      m.predict(base_logits,[tokens..]) -> corrected logits ; m.bytes() -> payload size"""
    def add(s,tokens,label,base_logits):
        b=np.asarray(base_logits,np.float32).ravel(); super().add(tokens,label,float(b[1]-b[0]))
    def add_anchor(s,tokens,label,base_logits):
        b=np.asarray(base_logits,np.float32).ravel(); super().add_anchor(tokens,label,float(b[1]-b[0]))
