"""Vectorised interleaved rANS (32-bit state, 16-bit renorm, 12-bit probs). numpy only. Decoder is ~25 lines."""
import numpy as np
PB=12; L=1<<16
def build_table(counts):
    c=np.asarray(counts,dtype=np.float64); n=len(c); f=np.maximum((c/c.sum()*(1<<PB)).astype(np.int64),(c>0).astype(np.int64))
    d=(1<<PB)-f.sum()
    while d!=0:
        i=int(np.argmax(c/f.clip(1))) if d>0 else int(np.argmax(np.where(f>1,f,0)))
        f[i]+=1 if d>0 else -1; d+= -1 if d>0 else 1
    return f
def encode(sym,tid,F,lanes=512):
    """sym uint8 (N,), tid per-symbol table id (N,) ; F [T,K] freq. returns (bytes words uint16, final states uint32 (lanes))"""
    N=len(sym); T=-(-N//lanes); pad=T*lanes-N
    sym=np.concatenate([sym,np.zeros(pad,np.uint8)]).reshape(T,lanes); tid=np.concatenate([tid,np.zeros(pad,tid.dtype)]).reshape(T,lanes)
    C=np.concatenate([np.zeros((F.shape[0],1),np.int64),np.cumsum(F,1)],1)
    x=np.full(lanes,L,dtype=np.uint64); chunks=[]
    for t in range(T-1,-1,-1):
        s=sym[t].astype(np.int64); i=tid[t].astype(np.int64); f=F[i,s].astype(np.uint64); st=C[i,s].astype(np.uint64)
        xm=((np.uint64(L)>>np.uint64(PB))<<np.uint64(16))*f; mk=x>=xm
        chunks.append((x[mk]&np.uint64(0xffff)).astype(np.uint16)); x=np.where(mk,x>>np.uint64(16),x)
        x=((x//f)<<np.uint64(PB))+(x%f)+st
    return np.concatenate(chunks[::-1]) if chunks else np.zeros(0,np.uint16), x.astype(np.uint32)
def decode(words,x0,tid,F,N,lanes=512):
    T=-(-N//lanes); pad=T*lanes-N; tid=np.concatenate([tid,np.zeros(pad,tid.dtype)]).reshape(T,lanes)
    C=np.concatenate([np.zeros((F.shape[0],1),np.int64),np.cumsum(F,1)],1); K=F.shape[1]
    # slot->symbol lookup per table
    LUT=np.zeros((F.shape[0],1<<PB),np.uint8)
    for i in range(F.shape[0]):
        for s in range(K):
            LUT[i,C[i,s]:C[i,s+1]]=s
    x=x0.astype(np.uint64); out=np.zeros((T,lanes),np.uint8); p=0; msk=np.uint64((1<<PB)-1)
    for t in range(T):
        i=tid[t].astype(np.int64); sl=(x&msk).astype(np.int64); s=LUT[i,sl]; out[t]=s
        s64=s.astype(np.int64); x=F[i,s64].astype(np.uint64)*(x>>np.uint64(PB))+sl.astype(np.uint64)-C[i,s64].astype(np.uint64)
        mk=x<np.uint64(L); k=int(mk.sum())
        if k: x[mk]=(x[mk]<<np.uint64(16))|words[p:p+k].astype(np.uint64); p+=k
    return out.reshape(-1)[:N]
if __name__=="__main__":
    import time
    rng=np.random.default_rng(0); N=3_000_000
    p=np.array([.1,.4,.4,.1]); sym=rng.choice(4,N,p=p).astype(np.uint8); tid=np.zeros(N,np.uint8)
    F=build_table(np.bincount(sym,minlength=4))[None]
    t=time.time(); w,x=encode(sym,tid,F); t1=time.time(); d=decode(w,x,tid,F,N); t2=time.time()
    H=-(p*np.log2(p)).sum()*N/8
    print(len(w)*2+len(x)*4,"bytes vs entropy",H,"ok",(d==sym).all(),t1-t,t2-t1)
