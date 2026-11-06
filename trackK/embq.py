"""Embedding compression for bert-base-uncased-SST-2. build_embedding(recipe)->(packed_bytes, lookup_fn).
recipe = dict(
  keep tiers by TRAIN frequency rank:  tiers=[dict(n=N, bits=4, g=128) | dict(n=N, pq=dict(ds=8,k=256))],
  fallback = 'nn' (nearest kept row by cosine) | 'unk' | 'mean',
  center=True (subtract mean row before quant), pos=N rows kept (fp16 or posbits), pos_bits=None|8..)
lookup_fn(ids LongTensor)->[...,768] fp32 word-emb ; lookup_fn.word_table() full 30522x768 ; lookup_fn.pos / .type
packed_bytes is a real byte string; lookup_fn is built ONLY from decoding it (decode(packed_bytes))."""
import sys, json, os, io, struct, collections, math, numpy as np, torch
sys.path.insert(0,os.path.expanduser('~/ten-percent/trackD'))
M="textattack/bert-base-uncased-SST-2"
V,D=30522,768
_cache={}
def base_tables():
    if 'b' not in _cache:
        from transformers import AutoModelForSequenceClassification as A
        m=A.from_pretrained(M).eval(); e=m.bert.embeddings
        _cache['b']=(e.word_embeddings.weight.detach().float().clone(),e.position_embeddings.weight.detach().float().clone(),e.token_type_embeddings.weight.detach().float().clone())
    return _cache['b']
def train_counts():
    if 'c' not in _cache:
        from lib import tok, data
        from datasets import load_dataset
        sp=json.load(open(os.path.expanduser("~/ten-percent/splits.json")))
        ex=set(sp['dev'])|set(sp['feedback'])  # exclude dev/feedback from counts -> dev is truly held-out
        tr=load_dataset("nyu-mll/glue","sst2")["train"]
        txt=[tr[i]['sentence'] for i in range(len(tr)) if i not in ex]
        cnt=np.zeros(V,dtype=np.int64)
        for i in range(0,len(txt),2000):
            for ids in tok(txt[i:i+2000],add_special_tokens=False)['input_ids']:
                for t in ids: cnt[t]+=1
        _cache['c']=cnt
    return _cache['c']
# ---------- bit packing
def pack_bits(codes,b):
    c=codes.astype(np.uint64).reshape(-1)
    if b==8: return c.astype(np.uint8).tobytes()
    bits=((c[:,None]>>np.arange(b,dtype=np.uint64))&1).astype(np.uint8).reshape(-1)
    return np.packbits(bits,bitorder='little').tobytes()
def unpack_bits(buf,b,n):
    if b==8: return np.frombuffer(buf,np.uint8)[:n].astype(np.int64)
    bits=np.unpackbits(np.frombuffer(buf,np.uint8),bitorder='little')[:n*b].reshape(n,b).astype(np.int64)
    return (bits<<np.arange(b)).sum(1)
def ent_bytes(codes,k):
    c=np.bincount(codes.reshape(-1),minlength=k).astype(float); p=c[c>0]/c.sum()
    return float(-(p*np.log2(p)).sum()*codes.size/8)
# ---------- group RTN
def rtn_enc(W,bits,g):
    N=W.shape[0]; Wg=W.reshape(N,-1,g); n=2**bits-1
    lo=Wg.min(2,keepdim=True)[0].half().float(); hi=Wg.max(2,keepdim=True)[0]
    s=((hi-lo).clamp(min=1e-8)/n).half().float().clamp(min=1e-8)
    q=((Wg-lo)/s).round().clamp(0,n)
    return q.reshape(N,-1).long().numpy(), s.reshape(N,-1).half().numpy(), lo.reshape(N,-1).half().numpy()
def rtn_dec(q,s,lo,g):
    N=q.shape[0]; return (torch.from_numpy(q).float().reshape(N,-1,g)*torch.from_numpy(s).float()[:,:,None]+torch.from_numpy(lo).float()[:,:,None]).reshape(N,-1)
# ---------- Lloyd-Max (non-uniform, global levels, per-group RMS scale)
def lm_enc(W,bits,g):
    N=W.shape[0]; Wg=W.reshape(N,-1,g); sc=Wg.pow(2).mean(2,keepdim=True).sqrt().half().float().clamp(min=1e-8)
    x=Wg/sc; k=2**bits; flat=x.reshape(-1); gen=torch.Generator().manual_seed(0)
    smp=flat[torch.randint(0,flat.numel(),(1500000,),generator=gen)]
    lv=torch.quantile(smp,(torch.arange(k)+0.5)/k)
    for _ in range(40):
        a=(smp[:,None]-lv[None]).abs().argmin(1)
        for j in range(k):
            m=a==j
            if m.any(): lv[j]=smp[m].mean()
    lv=lv.half().float().sort()[0]
    q=torch.cat([(c[:,None]-lv[None]).abs().argmin(1) for c in flat.split(2000000)])
    return q.reshape(N,-1).numpy(), sc.reshape(N,-1).half().numpy(), lv.half().numpy()
def lm_dec(q,sc,lv,g):
    N=q.shape[0]; lv=torch.from_numpy(lv).float()
    return (lv[torch.from_numpy(q).long()].reshape(N,-1,g)*torch.from_numpy(sc).float()[:,:,None]).reshape(N,-1)
# ---------- PQ
def kmeans_batched(X,k,iters=25,seed=0):
    # X [S,N,ds] -> centroids [S,k,ds], assign [S,N]
    g=torch.Generator().manual_seed(seed); S,N,ds=X.shape
    k=min(k,N); idx=torch.randperm(N,generator=g)[:k]; C=X[:,idx].clone()
    for it in range(iters):
        d=torch.cdist(X,C); a=d.argmin(2)
        oh=torch.zeros(S,N,k); oh.scatter_(2,a[:,:,None],1)
        cnt=oh.sum(1); newC=torch.einsum('snk,snd->skd',oh,X)/cnt.clamp(min=1)[:,:,None]
        empty=cnt==0
        if empty.any():  # reseed empties at random points
            r=torch.randint(0,N,(S,k),generator=g); newC=torch.where(empty[:,:,None],torch.gather(X,1,r[:,:,None].expand(-1,-1,ds)),newC)
        C=newC
    a=torch.cdist(X,C).argmin(2)
    return C,a
def pq_enc(W,ds,k):
    N=W.shape[0]; S=D//ds; X=W.reshape(N,S,ds).permute(1,0,2).contiguous()
    cs=[];as_=[]
    for i in range(0,S,8):
        C,a=kmeans_batched(X[i:i+8],k); cs.append(C); as_.append(a)
    return torch.cat(cs).half().numpy(), torch.cat(as_).T.contiguous().numpy()  # C[S,k,ds], a[N,S]
def pq_dec(C,a):
    C=torch.from_numpy(C).float(); a=torch.from_numpy(a).long(); S=C.shape[0]
    out=torch.stack([C[s][a[:,s]] for s in range(S)],1)  # N,S,ds
    return out.reshape(a.shape[0],-1)
# ---------- build
def build_embedding(recipe, seed_counts=None):
    Wt,P,T=base_tables(); cnt=train_counts() if seed_counts is None else seed_counts
    center=recipe.get('center',True); tiers=recipe['tiers']
    order=np.lexsort((np.arange(V),-cnt))           # by count desc
    special=[0,100,101,102,103]                      # PAD UNK CLS SEP MASK always kept (tier 0 member of first tier)
    order=np.array(special+[i for i in order if i not in special])
    tier_of=np.full(V,len(tiers),dtype=np.int64); pos=0; members=[]
    for ti,t in enumerate(tiers):
        ids=order[pos:pos+t['n']]; pos+=t['n']; tier_of[ids]=ti; members.append(np.sort(ids))
    kept=np.concatenate(members) if members else np.array([],int)
    mu=Wt.mean(0) if center else torch.zeros(D)
    # fallback map -> position in `kept` array (rows concatenated in tier order)
    kept_pos=np.full(V,-1); kept_pos[kept]=np.arange(len(kept))
    fb=recipe.get('fallback','nn')
    chunks=[]; hdr=dict(tiers=[],center=center,fallback=fb)
    chunks.append(mu.half().numpy().tobytes())
    ent=0.0
    for ti,t in enumerate(tiers):
        ids=members[ti]; W=(Wt[ids]-mu)
        e=dict(n=len(ids),**{k:v for k,v in t.items() if k!='n'})
        if 'pq' in t:
            C,a=pq_enc(W,t['pq']['ds'],t['pq']['k']); kk=C.shape[1]; b=max(1,math.ceil(math.log2(kk)))
            blob=C.tobytes()+pack_bits(a,b); e['pqk']=int(kk); ent+=len(C.tobytes())+ent_bytes(a,kk)
        elif t.get('lm'):
            q,sc,lv=lm_enc(W,t['bits'],t['g']); blob=lv.tobytes()+pack_bits(q,t['bits'])+sc.tobytes()
            ent+=lv.nbytes+ent_bytes(q,2**t['bits'])+sc.nbytes
        else:
            q,s,lo=rtn_enc(W,t['bits'],t['g']); blob=pack_bits(q,t['bits'])+s.tobytes()+lo.tobytes()
            ent+=ent_bytes(q,2**t['bits'])+s.nbytes+lo.nbytes
        e['len']=len(blob); chunks.append(blob); hdr['tiers'].append(e)
    # tier label per id (2-3 bits) and fallback target
    tb=max(1,math.ceil(math.log2(len(tiers)+1)))
    chunks.append(pack_bits(tier_of,tb)); hdr['tb']=tb
    nk=len(kept); fbits=max(1,math.ceil(math.log2(max(nk,2))))
    nonk=np.where(tier_of==len(tiers))[0]; hdr['nonk']=len(nonk)
    if fb=='nn' and len(nonk):
        # nearest kept (cosine on centered) ; decoded-domain kept rows used
        Kdec=torch.cat([_dec_tier(blob_i,hdr['tiers'][i],i,chunks) for i,blob_i in []],0) if False else None
        Wc=Wt-mu; Wn=torch.nn.functional.normalize(Wc,dim=1)
        kn=Wn[kept]; tgt=torch.empty(len(nonk),dtype=torch.long)
        for i in range(0,len(nonk),2048): tgt[i:i+2048]=(Wn[nonk[i:i+2048]]@kn.T).argmax(1)
        chunks.append(pack_bits(tgt.numpy(),fbits)); hdr['fbits']=fbits
    # positions / type
    pn=recipe.get('pos',512); pb=recipe.get('pos_bits')
    if pb:
        pm=P[:pn].mean(0); q,s,lo=rtn_enc(P[:pn]-pm,pb,recipe.get('pos_g',64)); pblob=pm.half().numpy().tobytes()+pack_bits(q,pb)+s.tobytes()+lo.tobytes()
    else: pblob=P[:pn].half().numpy().tobytes()
    hdr['pos']=dict(n=pn,bits=pb,g=recipe.get('pos_g',64),len=len(pblob)); chunks.append(pblob)
    chunks.append(T.half().numpy().tobytes())
    h=json.dumps(hdr).encode(); body=b''.join(chunks)
    packed=struct.pack('<I',len(h))+h+body
    packed_ent=len(packed)-sum(1 for _ in [0])  # placeholder
    lookup=decode(packed)
    lookup.ent_bytes=ent+len(packed)-sum(e['len'] for e in hdr['tiers'])
    return packed,lookup
def decode(packed):
    hl=struct.unpack('<I',packed[:4])[0]; hdr=json.loads(packed[4:4+hl]); buf=packed[4+hl:]; o=0
    mu=torch.from_numpy(np.frombuffer(buf[o:o+D*2],np.float16).copy()).float(); o+=D*2
    rows=[]
    for e in hdr['tiers']:
        blob=buf[o:o+e['len']]; o+=e['len']; n=e['n']
        if 'pq' in e:
            ds=e['pq']['ds']; S=D//ds; k=e['pqk']; cl=S*k*ds*2
            C=np.frombuffer(blob[:cl],np.float16).reshape(S,k,ds); b=max(1,math.ceil(math.log2(k)))
            a=unpack_bits(blob[cl:],b,n*S).reshape(n,S); rows.append(pq_dec(C,a)+mu)
        elif e.get('lm'):
            b,g=e['bits'],e['g']; ng=D//g; nl=2**b*2; nq=(n*D*b+7)//8
            lv=np.frombuffer(blob[:nl],np.float16); q=unpack_bits(blob[nl:nl+nq],b,n*D).reshape(n,D); sc=np.frombuffer(blob[nl+nq:],np.float16).reshape(n,ng)
            rows.append(lm_dec(q,sc,lv,g)+mu)
        else:
            b,g=e['bits'],e['g']; ng=D//g; nq=(n*D*b+7)//8
            q=unpack_bits(blob[:nq],b,n*D).reshape(n,D); s=np.frombuffer(blob[nq:nq+n*ng*2],np.float16).reshape(n,ng); lo=np.frombuffer(blob[nq+n*ng*2:],np.float16).reshape(n,ng)
            rows.append(rtn_dec(q,s,lo,g)+mu)
    K=torch.cat(rows,0)
    tb=hdr['tb']; nt=(V*tb+7)//8; tier_of=unpack_bits(buf[o:o+nt],tb,V); o+=nt
    nT=len(hdr['tiers']); kept=np.concatenate([np.where(tier_of==i)[0] for i in range(nT)]); nonk=np.where(tier_of==nT)[0]
    tab=torch.zeros(V,D); tab[kept]=K
    if len(nonk):
        if hdr['fallback']=='nn':
            fb=hdr['fbits']; nf=(len(nonk)*fb+7)//8; tgt=unpack_bits(buf[o:o+nf],fb,len(nonk)); o+=nf
            tab[nonk]=K[torch.from_numpy(tgt)]
        elif hdr['fallback']=='unk': tab[nonk]=tab[100].clone()
        else: tab[nonk]=mu
    p=hdr['pos']; pl=p['len']; pb=buf[o:o+pl]; o+=pl
    if p['bits']:
        n,b,g=p['n'],p['bits'],p['g']; pm=torch.from_numpy(np.frombuffer(pb[:D*2],np.float16).copy()).float(); ng=D//g; nq=(n*D*b+7)//8
        q=unpack_bits(pb[D*2:D*2+nq],b,n*D).reshape(n,D); r=D*2+nq
        s=np.frombuffer(pb[r:r+n*ng*2],np.float16).reshape(n,ng); lo=np.frombuffer(pb[r+n*ng*2:],np.float16).reshape(n,ng)
        pos=rtn_dec(q,s,lo,g)+pm
    else: pos=torch.from_numpy(np.frombuffer(pb,np.float16).reshape(p['n'],D).copy()).float()
    typ=torch.from_numpy(np.frombuffer(buf[o:o+2*D*2],np.float16).reshape(2,D).copy()).float()
    def lookup(ids): return tab[ids]
    lookup.word_table=lambda: tab; lookup.pos=pos; lookup.type=typ
    return lookup
