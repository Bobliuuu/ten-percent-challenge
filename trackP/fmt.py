"""Container v2 ("TPK2"): rANS-coded q + z symbols (static per-tensor tables), lzma'd misc blob (planar fp16 scales, small fp16 tensors,
embedding group ids, vocab, config). Reader returns (cfg, W state-dict, vocab) == trackM predict.read_model contract."""
import struct,json,lzma,numpy as np,torch
try: import rans
except ImportError: from . import rans
MAGIC=b"TPK2"
def _pl(a16):  # planar split of fp16 array -> bytes
    b=np.ascontiguousarray(a16).view(np.uint8).reshape(-1,2); return b[:,1].tobytes()+b[:,0].tobytes()
def _unpl(buf,n):
    hi=np.frombuffer(buf[:n],np.uint8); lo=np.frombuffer(buf[n:2*n],np.uint8); return np.stack([lo,hi],1).reshape(-1).view(np.float16)
def had(n):
    H=np.ones((1,1),np.float32)
    while len(H)<n: H=np.block([[H,H],[H,-H]])
    return torch.from_numpy(H)/n**0.5
def rot_mat(C,name,blk=256):
    gen=torch.Generator().manual_seed(sum(map(ord,name))%1000); sg=(torch.randint(0,2,(C,),generator=gen)*2-1).float()
    Hb=had(blk); R=torch.zeros(C,C)
    for i in range(0,C,blk): R[i:i+blk,i:i+blk]=Hb
    return sg[:,None]*R      # R = D*blockHadamard (orthogonal); stored weights are W@R, loader folds W_eff = dq @ R^T
def export(path,qd,small,gid,vocab_tokens,cfg,emb_name="bert.embeddings.word_embeddings.weight"):
    """qd: ordered dict name->(q,s,z,bits,g) ; embedding groups named emb_name+'#j' (rows = tokens with gid==j, ascending).
    small: dict name->fp16 numpy ; gid: uint8 array per vocab token."""
    names=list(qd); syms=[];tids=[];tabs=[];meta=[];scal=[]; lvs=[]
    for i,k in enumerate(names):
        q,s,z,b,g=qd[k][:5]; lm=z is None; qn=q.numpy().astype(np.uint8).ravel(); zn=np.zeros(0,np.uint8) if lm else z.numpy().astype(np.uint8).ravel()
        if lm: lvs.append(np.asarray(qd[k][5],np.float16))
        tabs.append(rans.build_table(np.bincount(qn,minlength=2**b))); tabs.append(rans.build_table(np.bincount(zn,minlength=2**b)+(1 if lm else 0)))
        syms+= [qn,zn]; tids+=[np.full(len(qn),2*i,np.uint16),np.full(len(zn),2*i+1,np.uint16)]
        scal.append(s.numpy().astype(np.float16).ravel()); meta.append(dict(n=k,shape=list(q.shape),b=b,g=g,lm=int(lm)))
    sym=np.concatenate(syms); tid=np.concatenate(tids); K=max(len(t) for t in tabs)
    F=np.zeros((len(tabs),K),np.int64)
    for i,t in enumerate(tabs): F[i,:len(t)]=t
    words,x0=rans.encode(sym,tid,F)
    sc=np.concatenate(scal); misc=[_pl(sc)]; sm=[]
    for k,v in small.items(): sm.append(dict(n=k,shape=list(v.shape))); misc.append(v.astype(np.float16).tobytes())
    for lv in lvs: misc.append(lv.tobytes())
    misc.append(np.asarray(gid,np.uint8).tobytes()); vt=("\n".join(vocab_tokens)).encode(); misc.append(vt)
    ftab=b"".join(np.asarray(t,np.uint16).tobytes() for t in tabs)
    hdr=dict(rot=256,q=meta,small=sm,nsc=len(sc),nsym=len(sym),nwords=len(words),vocab_len=len(vt),V=len(gid),cfg=cfg,tab_len=[len(t) for t in tabs],lanes=512)
    hz=lzma.compress(json.dumps(hdr,separators=(",",":")).encode(),preset=9|lzma.PRESET_EXTREME)
    mz=lzma.compress(b"".join(misc)+ftab,preset=9|lzma.PRESET_EXTREME)
    with open(path,"wb") as f:
        f.write(MAGIC+struct.pack("<III",len(hz),len(mz),len(words))); f.write(hz); f.write(mz); f.write(x0.tobytes()); f.write(words.tobytes())
    import os; return os.path.getsize(path)
def read(path,emb_name="bert.embeddings.word_embeddings.weight"):
    raw=open(path,"rb").read(); assert raw[:4]==MAGIC; lh,lm,nw=struct.unpack("<III",raw[4:16]); p=16
    h=json.loads(lzma.decompress(raw[p:p+lh])); p+=lh; mi=lzma.decompress(raw[p:p+lm]); p+=lm
    lanes=h["lanes"]; x0=np.frombuffer(raw[p:p+4*lanes],np.uint32); p+=4*lanes; words=np.frombuffer(raw[p:p+2*nw],np.uint16)
    q=h["q"]; nsc=h["nsc"]; o=2*nsc; sc=_unpl(mi,nsc).astype(np.float32)
    W={}
    for d in h["small"]:
        n=int(np.prod(d["shape"])); W[d["n"]]=torch.from_numpy(np.frombuffer(mi[o:o+2*n],np.float16).astype(np.float32).reshape(d["shape"]).copy()); o+=2*n
    for d in q:
        if d["lm"]: d["lv"]=np.frombuffer(mi[o:o+2*2**d["b"]],np.float16).astype(np.float32); o+=2*2**d["b"]
    V=h["V"]; gid=np.frombuffer(mi[o:o+V],np.uint8); o+=V; vocab=mi[o:o+h["vocab_len"]].decode().split("\n"); o+=h["vocab_len"]
    tl=h["tab_len"]; K=max(tl); F=np.zeros((len(tl),K),np.int64); fb=np.frombuffer(mi[o:],np.uint16); c=0
    for i,L in enumerate(tl): F[i,:L]=fb[c:c+L]; c+=L
    tid=np.concatenate([np.full(int(np.prod(d["shape"])),2*i+j,np.uint16) for i,d in enumerate(q) for j in (0,1)] if False else
        [np.full(n_,t_,np.uint16) for i,d in enumerate(q) for (n_,t_) in ((int(np.prod(d["shape"])),2*i),(0 if d["lm"] else int(np.prod(d["shape"]))//d["g"],2*i+1))])
    sym=rans.decode(words,x0,tid,F,h["nsym"],lanes); so=0; sco=0; parts={}
    for d in q:
        O,I=d["shape"]; N=O*I; ng=N//d["g"]; qq=sym[so:so+N].astype(np.float32).reshape(O,I//d["g"],d["g"]); so+=N
        ss=sc[sco:sco+ng].reshape(O,I//d["g"],1); sco+=ng
        if d["lm"]: parts[d["n"]]=torch.from_numpy((d["lv"][qq.astype(np.int64)]*ss).reshape(O,I).copy())+W["emb_mu"]
        else:
            zz=sym[so:so+ng].astype(np.float32).reshape(O,I//d["g"],1); so+=ng; parts[d["n"]]=torch.from_numpy(((qq-zz)*ss).reshape(O,I).copy())
    E=None
    for k,v in parts.items():
        if "#" in k:
            j=int(k.split("#")[1]); 
            if E is None: E=torch.zeros(V,v.shape[1])
            E[torch.from_numpy(gid==j)]=v
        else: W[k]=v@rot_mat(v.shape[1],k).T if (h.get("rot") and k.startswith("bert.encoder")) else v
    W.pop('emb_mu',None)
    if E is not None: W[emb_name]=E
    return h["cfg"],W,vocab
