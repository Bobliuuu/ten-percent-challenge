"""Embedding-table recipe with FULL vocab coverage, no gradient training.
quantize_embedding(E, freq, ...) -> (E_dequant, bytes). importable by trackA/final.py.
Tiers by train-frequency: hot rows (freq>=hot_thr) at hot_bits RTN g64; seen rows at mid_bits; unseen rows by `rare`:
  'rtnK' : K-bit RTN g64 ; 'pqM' : product quantization with M subspaces, 256 centroids (k-means, no gradients);
  'sign' : 1-bit sign x per-row fp16 scale ; 'lowrank_r' : shared rank-r basis (SVD of rare rows) + 4bit coeffs.
Bytes are counted exactly (scales/zeros fp16, codebooks fp16, tier bitmap 1 bit/row)."""
import torch
def rtn(w,bits,g=64):
    sh=w.shape; w2=w.reshape(-1,g); lo=w2.min(1,keepdim=True)[0]; hi=w2.max(1,keepdim=True)[0]
    n=2**bits-1; s=(hi-lo).clamp(min=1e-8)/n
    return (((w2-lo)/s).round().clamp(0,n)*s+lo).reshape(sh)
def rtn_bytes(nrows,H,bits,g=64): return nrows*H*bits/8+(nrows*H/g)*4
def kmeans(x,k,it=8):
    c=x[torch.randperm(len(x),generator=torch.Generator().manual_seed(0))[:k].to(x.device)].clone()
    for _ in range(it):
        a=torch.cdist(x,c).argmin(1)
        for j in range(k):
            sel=x[a==j]
            if len(sel): c[j]=sel.mean(0)
    return c,torch.cdist(x,c).argmin(1)
def pq(x,M,k=256):
    N,H=x.shape; d=H//M; out=torch.empty_like(x)
    for i in range(M):
        c,a=kmeans(x[:,i*d:(i+1)*d].contiguous(),min(k,N)); out[:,i*d:(i+1)*d]=c[a]
    return out, N*M*1+M*k*d*2
def quantize_embedding(E,freq,hot_thr=1,hot_bits=4,mid_thr=1,mid_bits=4,rare="rtn2",g=64):
    V,H=E.shape; out=E.clone(); dev=E.device; freq=freq.to(dev)
    hot=freq>=hot_thr; rare_m=~hot
    nb=V/8  # tier bitmap
    out[hot]=rtn(E[hot],hot_bits,g); nb+=rtn_bytes(int(hot.sum()),H,hot_bits,g)
    X=E[rare_m]; n=len(X)
    if rare.startswith("rtn"):
        b=int(rare[3:]); out[rare_m]=rtn(X,b,g); nb+=rtn_bytes(n,H,b,g)
    elif rare.startswith("pq"):
        o,by=pq(X,int(rare[2:])); out[rare_m]=o; nb+=by
    elif rare=="sign":
        s=X.abs().mean(1,keepdim=True); out[rare_m]=X.sign()*s; nb+=n*H/8+n*2
    elif rare.startswith("lowrank"):
        r=int(rare.split("_")[1]); mu=X.mean(0,keepdim=True)
        U,S,Vh=torch.linalg.svd(X-mu,full_matrices=False); B=Vh[:r]; C=(X-mu)@B.T
        out[rare_m]=rtn(C,4,r if r<=64 else 64)@B+mu if False else (rtn(C.reshape(-1,r),4,r).reshape(C.shape))@B+mu
        nb+=n*r*0.5+(n)*4+r*H*2+H*2
    elif rare=="mean": out[rare_m]=X.mean(0,keepdim=True); nb+=H*2
    return out,nb
SCHEMES=[("fp32",dict(hot_thr=0,rare="rtn2")),
 ("all4",dict(hot_thr=0)),
 ("seen4+unseen_rtn2 (trackA)",dict(rare="rtn2")),
 ("seen4+unseen_rtn3",dict(rare="rtn3")),
 ("seen4+unseen_rtn1ish_sign",dict(rare="sign")),
 ("seen4+unseen_pq96",dict(rare="pq96")),
 ("seen4+unseen_pq48",dict(rare="pq48")),
 ("seen4+unseen_pq24",dict(rare="pq24")),
 ("seen4+unseen_lowrank64",dict(rare="lowrank_64")),
 ("seen4+unseen_lowrank32",dict(rare="lowrank_32")),
 ("seen4+unseen_mean(unk-like)",dict(rare="mean")),
 ("seen3+unseen_pq48",dict(hot_bits=3,rare="pq48")),
 ("freq>=3:4b, rest pq48",dict(hot_thr=3,rare="pq48")),
 ("freq>=3:4b, rest rtn2",dict(hot_thr=3,rare="rtn2")),
 ("freq>=10:4b, rest pq48",dict(hot_thr=10,rare="pq48")),
 ("seen3+unseen_rtn2",dict(hot_bits=3,rare="rtn2")),
 ("seen2+unseen_pq48",dict(hot_bits=2,rare="pq48")),
]
