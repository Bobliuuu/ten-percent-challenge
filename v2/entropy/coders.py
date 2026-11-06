"""Lossless coding experiments on quantized-index tensors. numba kernels.
All models are adaptive (no side info) unless stated. Ideal cost = -log2 p under the adaptive model; real range coder verifies byte-exact decode."""
import numpy as np, numba as nb, math

@nb.njit(cache=True)
def adapt_cost(sym, ctx, nctx, L, inc, limit):
    cnt=np.ones((nctx,L)); tot=np.full(nctx,float(L)); bits=0.0
    for i in range(sym.shape[0]):
        c=ctx[i]; s=sym[i]
        bits-=math.log2(cnt[c,s]/tot[c])
        cnt[c,s]+=inc; tot[c]+=inc
        if tot[c]>limit:
            t=0.0
            for k in range(L):
                cnt[c,k]*=0.5; t+=cnt[c,k]
            tot[c]=t
    return bits

@nb.njit(cache=True)
def mix_cost(sym, ctxs, nctxs, L, inc, limit, lr, wctx, nw):
    """logistic (geometric) mixing of K adaptive models; weights selected by wctx. ctxs: (K,N)."""
    K=ctxs.shape[0]
    cnts=[np.ones((nctxs[k],L)) for k in range(K)]
    tots=[np.full(nctxs[k],float(L)) for k in range(K)]
    W=np.full((nw,K),1.0/K*1.3)
    bits=0.0; lp=np.zeros((K,L)); pm=np.zeros(L)
    for i in range(sym.shape[0]):
        s=sym[i]; w=W[wctx[i]]
        mx=-1e30
        for l in range(L):
            a=0.0
            for k in range(K):
                c=ctxs[k,i]; v=math.log(cnts[k][c,l]/tots[k][c]); lp[k,l]=v; a+=w[k]*v
            pm[l]=a
            if a>mx: mx=a
        z=0.0
        for l in range(L):
            pm[l]=math.exp(pm[l]-mx); z+=pm[l]
        for l in range(L): pm[l]/=z
        bits-=math.log2(pm[s])
        for k in range(K):
            e=lp[k,s]
            for l in range(L): e-=pm[l]*lp[k,l]
            w[k]+=lr*e
        for k in range(K):
            c=ctxs[k,i]; cnts[k][c,s]+=inc; tots[k][c]+=inc
            if tots[k][c]>limit:
                t=0.0
                for l in range(L):
                    cnts[k][c,l]*=0.5; t+=cnts[k][c,l]
                tots[k][c]=t
    return bits

# ---------------- real range coder (LZMA-style carry), integer adaptive model
@nb.njit(cache=True)
def rc_encode(sym, ctx, nctx, L, inc, limit):
    cnt=np.ones((nctx,L),dtype=np.int64); tot=np.full(nctx,L,dtype=np.int64)
    out=np.empty(sym.shape[0]*2+64,dtype=np.uint8); n=0
    low=np.int64(0); rng=np.int64(0xFFFFFFFF); cache=np.int64(0); csize=np.int64(1)
    for i in range(sym.shape[0]):
        c=ctx[i]; s=sym[i]
        start=0
        for k in range(s): start+=cnt[c,k]
        r=rng//tot[c]; low+=r*start; rng=r*cnt[c,s]
        while rng<(1<<24):
            rng<<=8
            # shift low
            if low<0xFF000000 or low>=(1<<32):
                carry=low>>32
                t=cache
                while True:
                    out[n]=(t+carry)&0xFF; n+=1; t=0xFF; csize-=1
                    if csize==0: break
                cache=(low>>24)&0xFF
            csize+=1
            low=(low&0x00FFFFFF)<<8
        cnt[c,s]+=inc; tot[c]+=inc
        if tot[c]>limit:
            t2=0
            for k in range(L):
                cnt[c,k]=(cnt[c,k]+1)>>1; t2+=cnt[c,k]
            tot[c]=t2
    for _ in range(5):
        if low<0xFF000000 or low>=(1<<32):
            carry=low>>32; t=cache
            while True:
                out[n]=(t+carry)&0xFF; n+=1; t=0xFF; csize-=1
                if csize==0: break
            cache=(low>>24)&0xFF
        csize+=1
        low=(low&0x00FFFFFF)<<8
    return out[:n]

@nb.njit(cache=True)
def rc_decode(buf, N, ctx, nctx, L, inc, limit):
    cnt=np.ones((nctx,L),dtype=np.int64); tot=np.full(nctx,L,dtype=np.int64)
    res=np.empty(N,dtype=np.uint8); p=1; rng=np.int64(0xFFFFFFFF); code=np.int64(0)
    for _ in range(4):
        code=(code<<8)|(buf[p] if p<buf.shape[0] else 0); p+=1
    for i in range(N):
        c=ctx[i]; r=rng//tot[c]; v=code//r
        if v>=tot[c]: v=tot[c]-1
        s=0; start=0
        while start+cnt[c,s]<=v:
            start+=cnt[c,s]; s+=1
        res[i]=s; code-=r*start; rng=r*cnt[c,s]
        while rng<(1<<24):
            rng<<=8; code=((code<<8)|(buf[p] if p<buf.shape[0] else 0))&0xFFFFFFFF; p+=1
        cnt[c,s]+=inc; tot[c]+=inc
        if tot[c]>limit:
            t2=0
            for k in range(L):
                cnt[c,k]=(cnt[c,k]+1)>>1; t2+=cnt[c,k]
            tot[c]=t2
    return res

def H(sym,L):
    h=np.bincount(sym,minlength=L).astype(np.float64); p=h[h>0]/h.sum(); return float(-(p*np.log2(p)).sum())

def make_ctx(Q,L,scales=None,prevQ=None,g=128):
    """Q: (R,C) uint8 indices. returns dict name->(ctx int32 flat, nctx). m = normalized |centered value| in [0,1]."""
    R,C=Q.shape; a=np.abs(2*Q.astype(np.int32)-(L-1)).astype(np.float32)/(L-1)
    out={}
    out["none"]=(np.zeros(R*C,np.int32),1)
    z=np.zeros_like(a)
    a1=z.copy(); a1[:,1:]=a[:,:-1]; a2=z.copy(); a2[:,2:]=a[:,:-2]
    rowb=np.minimum((a1+a2)*4,7).astype(np.int32)           # 8 bins (left neighbours)
    b1=z.copy(); b1[1:]=a[:-1]; b2=z.copy(); b2[2:]=a[:-2]
    colb=np.minimum((b1+b2)*4,7).astype(np.int32)           # 8 bins (rows above)
    cs=np.cumsum(a,axis=0)-a; n=np.arange(R,dtype=np.float32)[:,None]
    cm=np.where(n>0,cs/np.maximum(n,1),0.5)
    colh=np.minimum((cm*24),11).astype(np.int32)              # 12 bins running col mean
    pos=((np.arange(C)%g)//16).astype(np.int32)[None,:].repeat(R,0)  # 8 bins position in group
    # column history over a window of last 64 rows (more adaptive)
    f=lambda x:x.reshape(-1)
    out["row"]=(f(rowb),8); out["col"]=(f(colb),8); out["colhist"]=(f(colh),12); out["pos"]=(f(pos),8)
    out["colhist*row"]=(f(colh*8+rowb),96)
    out["colhist*row*col"]=(f((colh*8+rowb)*8+colb),768)
    out["colhist*pos"]=(f(colh*8+pos),96)
    if scales is not None:
        ls=np.log2(scales.astype(np.float32)); G=scales.shape[1]
        rel=ls-np.median(ls); sb=np.clip((rel*2+3),0,5).astype(np.int32)   # 6 bins
        sbf=np.repeat(sb,g,axis=1)[:,:C]
        out["scale"]=(f(sbf),6); out["colhist*row*scale"]=(f((colh*8+rowb)*6+sbf),576)
        # per-row mean of previous groups' |value|: row energy context
        rowcs=np.cumsum(a,axis=1)-a; rowm=rowcs/np.maximum(np.arange(C)[None,:],1)
        rowm[:,:8]=0.5; rm=np.minimum(rowm*24,11).astype(np.int32)
        out["rowhist"]=(f(rm),12); out["colhist*rowhist"]=(f(colh*12+rm),144)
        out["colhist*rowhist*row"]=(f((colh*12+rm)*8+rowb),1152)
    if prevQ is not None:
        ap=np.abs(2*prevQ.astype(np.int32)-(L-1)).astype(np.float32)/(L-1)
        pb=np.minimum(ap*8,7).astype(np.int32)
        out["prevlayer"]=(f(pb),8); out["prevlayer*colhist*row"]=(f((pb*12+colh)*8+rowb),768)
    return out
