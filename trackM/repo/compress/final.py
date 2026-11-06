"""Track A final: GPTQ mixed-precision (2/3/4-bit, group 128/256) + vocab pruning, real bit-packed export + reload.
Usage: python trackA/final.py [out.bin]   (uses only SST-2 train split: vocab/positions from train text, calibration+eval on dev idx)"""
import sys,os,json,struct,time,io
sys.argv=[sys.argv[0],"x"]+sys.argv[1:]
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from gq import *
from transformers import BertConfig, BertTokenizerFast
OUT=sys.argv[2] if len(sys.argv)>2 else HERE+"/model.bin"
names=dict(q="attention.self.query",k="attention.self.key",v="attention.self.value",a="attention.output.dense",i="intermediate.dense",o="output.dense")
inv={v:k for k,v in names.items()}
BITS=dict(q=2,k=2,v=3,a=3,i=[4]*6+[3]*6,o=3)       # config "Q"
def bitsfn(l,n):
    x=BITS[inv[n]]; return x[l] if isinstance(x,list) else x
gfn=lambda l,n: 256 if inv[n]=="o" else 128
EMB_BITS,EMB_REST_BITS,EMB_G,POOL_BITS=3,2,128,6   # train-seen tokens 3-bit, other tokens 2-bit (full vocab kept)
# ---------------- bit packing
def pack(q,b):
    q=q.numpy().astype(np.uint8).ravel(); bits=((q[:,None]>>np.arange(b))&1).astype(np.uint8).ravel(); return np.packbits(bits).tobytes()
def unpack(buf,b,n):
    bits=np.unpackbits(np.frombuffer(buf,dtype=np.uint8))[:n*b].reshape(n,b); return torch.from_numpy((bits<<np.arange(b)).sum(1).astype(np.float32))
def export(m,packed,path,vocab_tokens,mask):
    qkeys={"emb_a":"bert.embeddings.word_embeddings.weight#a","emb_b":"bert.embeddings.word_embeddings.weight#b","pool":"bert.pooler.dense.weight"}
    for l in range(12):
        for n in LIN: qkeys[f"L{l}.{n}"]=f"bert.encoder.layer.{l}.{n}.weight"
    blobs=[];meta={"q":{},"fp16":{}};off=0
    def add(b):
        nonlocal off; blobs.append(b); o=off; off+=len(b); return o
    for k,(q,s,z,b,g) in packed.items():
        pn=pack(q,b); o1=add(pn); o2=add(s.half().numpy().tobytes()); o3=add(z.half().numpy().tobytes())
        meta["q"][qkeys[k]]=dict(shape=list(q.shape),bits=b,g=g,o=[o1,o2,o3])
    m=m.cpu()
    for n,p in m.state_dict().items():
        if n=="bert.embeddings.word_embeddings.weight" or n in meta["q"] or "position_ids" in n or "token_type_ids" in n.split(".")[-1] and p.dtype==torch.long: continue
        meta["fp16"][n]=dict(shape=list(p.shape),o=add(p.detach().cpu().half().numpy().tobytes()))
    meta["mask_o"]=add(np.packbits(mask.numpy().astype(np.uint8)).tobytes())
    vt=("\n".join(vocab_tokens)).encode(); import zlib; vz=zlib.compress(vt,9)
    meta["vocab_o"]=add(vz); meta["vocab_len"]=len(vz)
    meta["cfg"]=m.config.to_dict()
    hj=json.dumps(meta,separators=(",",":")).encode()
    with open(path,"wb") as f: f.write(struct.pack("<I",len(hj))); f.write(hj); [f.write(b) for b in blobs]
    return os.path.getsize(path)
def load(path):
    import zlib
    raw=open(path,"rb").read(); n=struct.unpack("<I",raw[:4])[0]; meta=json.loads(raw[4:4+n]); data=raw[4+n:]
    cfg=BertConfig.from_dict(meta["cfg"]); cfg.max_position_embeddings=meta["fp16"]["bert.embeddings.position_embeddings.weight"]["shape"][0]; m=AutoModelForSequenceClassification.from_config(cfg).eval(); sd={}
    for k,d in meta["q"].items():
        O,I=d["shape"]; b=d["bits"]; g=d["g"]; o1,o2,o3=d["o"]
        nb=(O*I*b+7)//8; q=unpack(data[o1:o1+nb],b,O*I).reshape(O,I)
        ng=O*(I//g); s=torch.from_numpy(np.frombuffer(data[o2:o2+2*ng],dtype=np.float16).astype(np.float32).copy()).reshape(O,I//g)
        z=torch.from_numpy(np.frombuffer(data[o3:o3+2*ng],dtype=np.float16).astype(np.float32).copy()).reshape(O,I//g)
        if 'word_embeddings' in k: sd.setdefault("_parts",{})[k[-1]]=dequant(q,s,z,g)
        else: sd[k]=dequant(q,s,z,g)
    parts=sd.pop("_parts"); V=meta["cfg"]["vocab_size"]
    mask=torch.from_numpy(np.unpackbits(np.frombuffer(data[meta["mask_o"]:meta["mask_o"]+(V+7)//8],dtype=np.uint8))[:V].astype(bool))
    E=torch.zeros(V,768); ps=[parts['a'],parts['b']]
    E[mask]=ps[0]; E[~mask]=ps[1]; sd["bert.embeddings.word_embeddings.weight"]=E
    for k,d in meta["fp16"].items():
        cnt=int(np.prod(d["shape"])); sd[k]=torch.from_numpy(np.frombuffer(data[d["o"]:d["o"]+2*cnt],dtype=np.float16).astype(np.float32).copy()).reshape(d["shape"])
    missing=m.load_state_dict(sd,strict=False); 
    vocab=zlib.decompress(data[meta["vocab_o"]:meta["vocab_o"]+meta["vocab_len"]]).decode().split("\n")
    return m,vocab,missing
def tok_from_vocab(vocab):
    p=HERE+"/vocab_pruned.txt"; open(p,"w").write("\n".join(vocab)); return BertTokenizerFast(p,do_lower_case=True)
if __name__=="__main__":
    t0=time.time()
    import gq
    ID=torch.arange(tok.vocab_size); gq.CB=encode(tx[:1000],ID); gq.B=encode(tx,ID)   # FULL vocab kept (no id remap)
    mask=torch.zeros(tok.vocab_size,dtype=torch.bool); mask[torch.tensor(keep)]=True   # keep = ids seen in SST-2 train text
    full=prune_vocab(base_model(),list(range(tok.vocab_size)),64)   # only trims position table to 64
    m,pk=gptq_quantize(full,bitsfn,gfn,embbits=EMB_BITS,gemb=EMB_G,poolbits=POOL_BITS,clip=True,embrest=(mask,EMB_REST_BITS))
    print(gq.rep2(m,"fake-quant"),f"{time.time()-t0:.0f}s")
    vtoks=[tok.convert_ids_to_tokens(i) for i in range(tok.vocab_size)]
    size=export(m,pk,OUT,vtoks,mask)
    base_bytes=sum(p.numel() for p in base.parameters())*4
    print(f"EXPORT {OUT}: {size} bytes = {size/1e6:.2f} MB ; fp32 base {base_bytes/1e6:.1f} MB ; ratio {size/base_bytes*100:.2f}%")
    m2,vocab,missing=load(OUT); print("load report: missing",missing.missing_keys,"unexpected",missing.unexpected_keys)
    # logits-identical check (same id remap path)
    l1=logits(m,gq.B); l2=logits(m2,gq.B); print("max |logit diff| fake vs reloaded:",float((l1-l2).abs().max()))
    print(gq.rep2(m2,"reloaded"))
    # fully self-contained path: tokenizer built from pruned vocab only
    tk=tok_from_vocab(vocab); out=[]
    m2=m2.to(DEV)
    with torch.no_grad():
        for i in range(0,len(tx),64):
            b=tk(tx[i:i+64],padding=True,truncation=True,max_length=64,return_tensors="pt")
            out.append(m2(**{k:v.to(DEV) for k,v in b.items()}).logits.cpu())
    p=torch.cat(out).argmax(-1).numpy(); a=p==y;ag=p==BP
    print(f"SELF-CONTAINED reload (pruned-vocab tokenizer): devacc={a.mean():.4f} agree={ag.mean():.4f} heldout(dev[1000:]) acc={a[1000:].mean():.4f} agree={ag[1000:].mean():.4f}")
    print(f"base dev acc {np.mean(BP==y):.4f}")
    np_=sum(p.numel() for p in m2.parameters()); print(f"params kept {np_} vs {sum(p.numel() for p in base.parameters())} = {np_/sum(p.numel() for p in base.parameters())*100:.2f}% ; bits/param effective {size*8/np_:.2f}")

    # ---- dev2 report (coordinator proxy): acc / agreement with base, self-contained path
    sys.path.insert(0,'/Users/jhome/ten-percent/trackG'); from dev2 import load_dev2
    bm=base.to(DEV)
    def run(model,tk,texts):
        o=[]
        with torch.no_grad():
            for i in range(0,len(texts),64):
                b=tk(texts[i:i+64],padding=True,truncation=True,max_length=64,return_tensors="pt")
                o.append(model(**{k:v.to(DEV) for k,v in b.items()}).logits.cpu())
        return torch.cat(o).argmax(-1).numpy()
    for nm,(tt,ll) in {"dev2":load_dev2(),"dev2_hard":load_dev2(hard=True),"old_dev":(tx,list(y))}.items():
        ll=np.array(ll); pb=run(bm,tok,tt); pq=run(m2,tk,tt)
        print(f"REPORT {nm}: n={len(tt)} base_acc={np.mean(pb==ll):.4f} quant_acc={np.mean(pq==ll):.4f} agree={np.mean(pb==pq):.4f}")
