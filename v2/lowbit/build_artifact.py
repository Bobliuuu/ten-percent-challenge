"""Build + verify a packed ternary artifact from a packed BitNet-style HF checkpoint (per-tensor scale, uint8 2-bit packed).
Counts EVERY byte of: entropy-coded ternary payload (+prob tables), quantized kept-vocab embeddings (+scales/zeros),
fp16 letter head rows, kept-id list, norms/scales, tokenizer (xz), config. Then DECODES the artifact only, rebuilds the
HF model from decoded bytes (no original weights), and scores MMLU dev + MDC dev.
Usage: CFG='{"model":"microsoft/bitnet-b1.58-2B-4T","V":32768,"ebits":4,"g":64}' modal run --detach build_artifact.py"""
import modal, json, os
app = modal.App("tenpct-lowbit-build")
img = (modal.Image.debian_slim().pip_install("torch","transformers==4.57.1","numpy","safetensors","huggingface_hub","constriction","accelerate")
       .add_local_file("mmlu_dev.jsonl","/root/mmlu_dev.jsonl").add_local_file("mdc_dev.jsonl","/root/mdc_dev.jsonl")
       .add_local_dir("vocabsrc","/root/vocabsrc"))
vol = modal.Volume.from_name("lowbit-data", create_if_missing=True)
hf = modal.Volume.from_name("hf-cache", create_if_missing=True)
def fmt_mmlu(r):
    s="The following are multiple choice questions (with answers) about "+r["subject"].replace("_"," ")+".\n\n"+r["question"]+"\n"
    for l,c in zip("ABCD",r["choices"]): s+=f"{l}. {c}\n"
    return s+"Answer:"
def fmt_mdc(r):
    s=r["inst"]+"\n\nText: "+r["text"]+"\n\nOptions:\n"
    for l,c in zip("ABCDEFGHIJKLMN",r["names"]): s+=f"{l}. {c}\n"
    return s+"Answer:"

def probs_from_counts(cnt):
    """cnt [...,3] float -> quantized probs (uint16 q0,q1 stored; q2 = 65536-q0-q1), min 1/65536 each. returns (q uint16 [...,2], probs float32 [...,3])"""
    import numpy as np
    p=cnt/np.maximum(cnt.sum(-1,keepdims=True),1)
    q=np.clip(np.round(p*65536),1,65534).astype(np.int64)
    q0=q[...,0]; q1=q[...,1]
    over=(q0+q1)>65534; 
    q1=np.where(over,65535-q0-1+0,q1); q1=np.maximum(q1,1)
    return np.stack([q0,q1],-1).astype(np.uint16)
def probs_from_q(q):
    import numpy as np
    q=q.astype(np.int64); q2=65536-q[...,0]-q[...,1]
    return (np.stack([q[...,0],q[...,1],q2],-1)/65536.0).astype(np.float32)

def encode_matrix(t, cons):
    """t int8 [R,C] in {-1,0,1}. choose mode g/r/c with exact cost; returns (mode, qtable bytes, payload bytes(np.uint32 words))"""
    import numpy as np, constriction
    R,C=t.shape; s=(t+1).astype(np.int32)
    def tabs(mode):
        if mode=="g": cnt=np.bincount(s.ravel(),minlength=3).astype(np.float64)[None]
        elif mode=="r": cnt=np.stack([(s==k).sum(1) for k in range(3)],-1).astype(np.float64)
        else: cnt=np.stack([(s==k).sum(0) for k in range(3)],-1).astype(np.float64)
        return probs_from_counts(cnt)
    best=None
    for mode in "grc":
        q=tabs(mode); pr=probs_from_q(q).astype(np.float64)
        if mode=="g": P=pr[0][s]; 
        elif mode=="r": P=np.take_along_axis(pr,s,1)
        else: P=np.take_along_axis(pr.T,s.T,1).T if False else pr[np.arange(C)[None,:],s]
        bits=float(-np.log2(P).sum())+q.size*16
        if best is None or bits<best[0]: best=(bits,mode,q)
    _,mode,q=best; pr=probs_from_q(q)
    if mode=="g": per=np.broadcast_to(pr[0],(R*C,3)) ; 
    elif mode=="r": per=np.repeat(pr,C,axis=0)
    else: per=np.tile(pr,(R,1))
    coder=constriction.stream.stack.AnsCoder()
    coder.encode_reverse(s.ravel().astype(np.int32), constriction.stream.model.Categorical(perfect=False), np.ascontiguousarray(per))
    words=coder.get_compressed()
    return mode,q,words
def decode_matrix(mode,q,words,R,C):
    import numpy as np, constriction
    pr=probs_from_q(q)
    if mode=="g": per=np.broadcast_to(pr[0],(R*C,3))
    elif mode=="r": per=np.repeat(pr,C,axis=0)
    else: per=np.tile(pr,(R,1))
    coder=constriction.stream.stack.AnsCoder(words)
    s=coder.decode(constriction.stream.model.Categorical(perfect=False), np.ascontiguousarray(per))
    return (s.reshape(R,C).astype(np.int8)-1)

def pack_hf(t):
    import numpy as np
    R,C=t.shape; pr=R//4; u=(t+1).astype(np.uint8); p=np.zeros((pr,C),np.uint8)
    for i in range(4): p|=(u[i*pr:(i+1)*pr]<<(2*i))
    return p

@app.function(image=img, gpu="A100-80GB", cpu=8, memory=98304, timeout=14400, volumes={"/hf":hf,"/d":vol})
def run(cfg):
    import torch, numpy as np, time, io, lzma, struct, shutil
    os.environ["HF_HOME"]="/hf"
    from huggingface_hub import hf_hub_download, snapshot_download
    from safetensors import safe_open
    from safetensors.torch import save_file
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from transformers.integrations.bitnet import unpack_weights
    M=cfg["model"]; V=cfg.get("V",32768); eb=cfg.get("ebits",4); G=cfg.get("g",64); tag=cfg.get("tag",M.split("/")[-1])
    t0=time.time(); out=dict(cfg=cfg)
    src=snapshot_download(M,allow_patterns=["*.json","*.safetensors","*.txt","*.model"]); tok=AutoTokenizer.from_pretrained(M)
    A=f"/d/art/{tag}"; os.makedirs(A,exist_ok=True)
    f=safe_open(src+"/model.safetensors","pt"); keys=list(f.keys())
    emb_key="model.embed_tokens.weight"; head_key="lm_head.weight" if "lm_head.weight" in keys else None
    Wemb=f.get_tensor(emb_key); Vfull,H=Wemb.shape
    # fp16 bytes of the base: params*2 (logical params = unpacked count)
    lin=[k for k in keys if k.endswith("proj.weight")]
    n_lin=0
    for k in lin:
        s=f.get_slice(k).get_shape(); n_lin+=s[0]*4*s[1]
    n_other=sum(int(np.prod(f.get_slice(k).get_shape())) for k in keys if k not in lin)
    n_params=n_lin+n_other; base_fp16=n_params*2
    out.update(n_params=n_params,n_lin=n_lin,base_fp16_bytes=base_fp16,budget=base_fp16*0.1)
    # ---- vocab keep set from NON-LOCKED text
    cnt=np.zeros(Vfull,np.int64)
    srcs=[("mmlu_pool",fmt_mmlu),("mmlu_calib",fmt_mmlu),("mdc_pool",fmt_mdc),("mdc_calib",fmt_mdc)]
    for fn,fm in srcs:
        for line in open(f"/root/vocabsrc/{fn}.jsonl"):
            r=json.loads(line)
            if fn=="mmlu_calib" and "choices" not in r: continue
            try: ids=tok.encode(fm(r))
            except Exception: continue
            np.add.at(cnt,ids,1)
    letter_ids=[tok.encode(" "+l,add_special_tokens=False)[-1] for l in "ABCDEFGHIJKLMN"]
    vocab_inv=tok.convert_ids_to_tokens(list(range(Vfull))); 
    keep=set(np.argsort(-cnt)[:V].tolist())
    for t_,i in tok.get_vocab().items():
        if len(t_)==1 and i<Vfull: keep.add(i)
    keep|=set(letter_ids)|set(i for i in tok.all_special_ids if i<Vfull)
    keep=sorted(keep); nk=len(keep)
    out["nkeep"]=nk
    keepmask=np.zeros(Vfull,bool); keepmask[keep]=True
    old2new={o:n for n,o in enumerate(keep)}
    keepstr={vocab_inv[i]:i for i in keep}; maxl=max(len(k) for k in keepstr)
    oov=[0,0]
    def remap(ids):
        o=[]
        for i in ids:
            oov[1]+=1
            if keepmask[i]: o.append(old2new[i]); continue
            oov[0]+=1; t_=vocab_inv[i]; p=0
            while p<len(t_):
                for q in range(min(len(t_),p+maxl),p,-1):
                    if t_[p:q] in keepstr: o.append(old2new[keepstr[t_[p:q]]]); p=q; break
                else: o.append(old2new[tok.unk_token_id] if tok.unk_token_id in old2new else old2new[keep[0]]); p+=1
        return o
    # ---- embedding quant: asymmetric RTN, b bits, group G, fp16 scale+min per group
    def quant_emb(W):
        W=W.float(); N,Hh=W.shape; Wg=W.view(N,Hh//G,G)
        mn=Wg.amin(-1,keepdim=True); mx=Wg.amax(-1,keepdim=True); L=2**eb-1
        sc=((mx-mn)/L).clamp_min(1e-8); sc=sc.half().float(); mnh=mn.half().float()
        qi=torch.round((Wg-mnh)/sc).clamp(0,L).to(torch.uint8)
        return qi.view(N,Hh).numpy(),sc.squeeze(-1).half().numpy(),mnh.squeeze(-1).half().numpy()
    Wk=Wemb[torch.tensor(keep)]
    qi,sc,mn=quant_emb(Wk)
    # bit-pack qi
    def packbits(q,b):
        flat=q.ravel().astype(np.uint64)
        return np.packbits(((flat[:,None]>>np.arange(b-1,-1,-1,dtype=np.uint64))&1).astype(np.uint8).ravel()).tobytes()
    emb_bytes=packbits(qi,eb); 
    # ---- ternary payload
    tern_blob=io.BytesIO(); meta=[]; nbits_pay=0
    for k in lin:
        u=f.get_tensor(k); t=unpack_weights(u,torch.bfloat16).to(torch.int8).numpy()
        mode,q,words=encode_matrix(t,None)
        meta.append(dict(k=k,R=t.shape[0],C=t.shape[1],mode=mode,qoff=tern_blob.tell()))
        tern_blob.write(q.tobytes()); meta[-1]["woff"]=tern_blob.tell(); meta[-1]["nw"]=len(words); tern_blob.write(words.tobytes())
        if len(meta)%20==0: print("enc",len(meta),len(lin),tern_blob.tell(),flush=True)
    tern=tern_blob.getvalue()
    # ---- small tensors (bf16 raw): everything not lin, not embed/head
    small=[k for k in keys if k not in lin and k not in (emb_key,head_key)]
    sbuf=io.BytesIO(); smeta=[]
    for k in small:
        t=f.get_tensor(k); smeta.append(dict(k=k,dtype=str(t.dtype),shape=list(t.shape),off=sbuf.tell()))
        sbuf.write(t.view(torch.int16).numpy().tobytes() if t.dtype in (torch.bfloat16,torch.float16) else t.numpy().tobytes())
    smalls=sbuf.getvalue()
    headrows=(f.get_tensor(head_key) if head_key else Wemb)[torch.tensor(letter_ids)]
    head_bytes=headrows.view(torch.int16).numpy().tobytes()
    keep_bytes=np.array(keep,dtype=np.uint32 if Vfull>65535 else np.uint16).tobytes()
    # tokenizer files (xz)
    tokfiles={}
    for fn in ("tokenizer.json","tokenizer_config.json","special_tokens_map.json","tokenizer.model","generation_config.json"):
        if os.path.exists(src+"/"+fn): tokfiles[fn]=open(src+"/"+fn,"rb").read()
    cfgjson=open(src+"/config.json","rb").read()
    tok_xz=lzma.compress(b"".join(struct.pack("<I",len(n))+n.encode()+struct.pack("<I",len(v))+v for n,v in tokfiles.items()),preset=9|lzma.PRESET_EXTREME)
    cfg_xz=lzma.compress(cfgjson,preset=9|lzma.PRESET_EXTREME)
    meta_json=lzma.compress(json.dumps(dict(meta=meta,smeta=smeta,keep_dtype="u32" if Vfull>65535 else "u16",nkeep=nk,H=H,Vfull=Vfull,eb=eb,G=G,letter_ids=letter_ids,emb_key=emb_key,head_key=head_key,tied=head_key is None)).encode(),preset=9)
    files={"ternary.bin":tern,"emb.bin":emb_bytes,"emb_scale.bin":sc.tobytes(),"emb_min.bin":mn.tobytes(),"head_letters.bin":head_bytes,
           "keep_ids.bin":keep_bytes,"small.bin":smalls,"tokenizer.xz":tok_xz,"config.xz":cfg_xz,"meta.xz":meta_json}
    sizes={k:len(v) for k,v in files.items()}; total=sum(sizes.values())
    for k,v in files.items(): open(f"{A}/{k}","wb").write(v)
    out.update(file_bytes=sizes,total_bytes=total,ratio=total/base_fp16,ternary_bits_per_w=len(tern)*8/n_lin,modes={m:sum(1 for x in meta if x["mode"]==m) for m in "grc"})
    print("SIZES",json.dumps(out),flush=True); vol.commit()
    # ================= DECODE FROM ARTIFACT ONLY + VERIFY =================
    del f
    files={k:open(f"{A}/{k}","rb").read() for k in sizes}
    D=f"/tmp/dec_{tag}"; shutil.rmtree(D,ignore_errors=True); os.makedirs(D)
    mj=json.loads(lzma.decompress(files["meta.xz"]))
    sd={}; exact=True
    tb=np.frombuffer(files["ternary.bin"],dtype=np.uint8)
    orig=safe_open(src+"/model.safetensors","pt")
    for m in mj["meta"]:
        q=np.frombuffer(tb[m["qoff"]:m["woff"]].tobytes(),dtype=np.uint16).reshape(-1,2)
        words=np.frombuffer(tb[m["woff"]:m["woff"]+4*m["nw"]].tobytes(),dtype=np.uint32)
        t=decode_matrix(m["mode"],q,words,m["R"],m["C"]); p=pack_hf(t)
        ok=bool((torch.from_numpy(p)==orig.get_tensor(m["k"])).all()); exact&=ok  # VERIFICATION ONLY (original used just to check)
        sd[m["k"]]=torch.from_numpy(p)
    out["ternary_roundtrip_exact"]=exact; print("roundtrip exact",exact,flush=True)
    sb=files["small.bin"]
    for s in mj["smeta"]:
        dt=getattr(torch,s["dtype"].split(".")[-1]); n=int(np.prod(s["shape"])); isz=torch.tensor([],dtype=dt).element_size()
        sd[s["k"]]=torch.frombuffer(bytearray(sb[s["off"]:s["off"]+n*isz]),dtype=dt).reshape(s["shape"]).clone()
    # embeddings dequant from artifact bytes
    eb_,G_,nk_,H_=mj["eb"],mj["G"],mj["nkeep"],mj["H"]
    bits=np.unpackbits(np.frombuffer(files["emb.bin"],dtype=np.uint8))[:nk_*H_*eb_].reshape(-1,eb_)
    qv=np.zeros(bits.shape[0],np.uint8)
    for j in range(eb_): qv=(qv<<1)|bits[:,j]
    qv=torch.from_numpy(qv.reshape(nk_,H_//G_,G_)).float()
    scd=torch.from_numpy(np.frombuffer(files["emb_scale.bin"],dtype=np.float16).reshape(nk_,H_//G_,1).copy()).float()
    mnd=torch.from_numpy(np.frombuffer(files["emb_min.bin"],dtype=np.float16).reshape(nk_,H_//G_,1).copy()).float()
    emb_deq=(qv*scd+mnd).reshape(nk_,H_).to(torch.bfloat16)
    keep_dec=np.frombuffer(files["keep_ids.bin"],dtype=np.uint32 if mj["keep_dtype"]=="u32" else np.uint16).tolist()
    assert keep_dec==keep
    letters=torch.from_numpy(np.frombuffer(files["head_letters.bin"],dtype=np.int16).reshape(len(letter_ids),H_).copy()).view(torch.bfloat16)
    # tokenizer from artifact
    raw=lzma.decompress(files["tokenizer.xz"]); p=0
    while p<len(raw):
        n=struct.unpack("<I",raw[p:p+4])[0]; p+=4; nm=raw[p:p+n].decode(); p+=n
        n=struct.unpack("<I",raw[p:p+4])[0]; p+=4; open(f"{D}/{nm}","wb").write(raw[p:p+n]); p+=n
    cj=json.loads(lzma.decompress(files["config.xz"])); cj["vocab_size"]=nk_
    json.dump(cj,open(f"{D}/config.json","w"))
    sd[mj["emb_key"]]=emb_deq
    if mj["head_key"]: sd[mj["head_key"]]=torch.zeros(nk_,H_,dtype=torch.bfloat16)  # unused placeholder (scoring uses head_letters.bin)
    save_file(sd,f"{D}/model.safetensors",metadata={"format":"pt"})
    tok2=AutoTokenizer.from_pretrained(D)
    model=AutoModelForCausalLM.from_pretrained(D,torch_dtype=torch.bfloat16).cuda().eval()
    def evalset(name,fn,fl,K,tag2):
        rows=[json.loads(x) for x in open(fl)]
        lg=np.full((len(rows),K),-1e9,dtype=np.float32); Wl=letters[:K].cuda()
        with torch.no_grad():
            for i,r in enumerate(rows):
                x=torch.tensor([remap(tok2.encode(fn(r)))[-1024:]]).cuda()
                h=model.model(input_ids=x).last_hidden_state[0,-1]
                full=(h@Wl.T).float().cpu().numpy(); k=len(r["choices"] if name=="mmlu" else r["names"]); lg[i,:k]=full[:k]
        y=np.array([r["answer"] for r in rows]); pred=lg.argmax(1)
        base=np.load(f"/d/{M.replace('/','_')}.{name}.npy").argmax(1)
        out[f"{tag2}_{name}_acc"]=float((pred==y).mean()); out[f"{tag2}_{name}_agree_base"]=float((pred==base).mean())
        out[f"base_{name}_acc"]=float((base==y).mean())
        np.save(f"/d/art/{tag}/logits_{tag2}_{name}.npy",lg)
    for tag2 in ("art","trim16"):
        if tag2=="trim16":
            model.model.embed_tokens.weight.data=Wk.to(torch.bfloat16).cuda(); letters=headrows.to(torch.bfloat16)
        evalset("mmlu",fmt_mmlu,"/root/mmlu_dev.jsonl",4,tag2); evalset("mdc",fmt_mdc,"/root/mdc_dev.jsonl",14,tag2)
        print("EVAL",tag2,{k:v for k,v in out.items() if k.startswith(tag2)},flush=True)
    out["oov_rate"]=oov[0]/max(1,oov[1]); out["secs"]=time.time()-t0
    json.dump(out,open(f"/d/art/{tag}/result.json","w"),indent=1); vol.commit()
    print("RESULT_JSON "+json.dumps(out)); return out
@app.local_entrypoint()
def main():
    for c in os.environ["CFGS"].split(";"): run.spawn(json.loads(c))
