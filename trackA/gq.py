import sys; sys.path.insert(0,'/Users/jhome/ten-percent/trackA')
from exp import *
NC=1000  # calibration = dev[:1000]
def calib_batches():
    return encode(tx[:NC],T)
CB=calib_batches()
@torch.no_grad()
def gptq_quantize(m,bitsfn,g,embbits=4,gemb=128,poolbits=6,damp=0.01,clip=False,embrest=None):
    """returns m (dequantized in place) and dict of packed tensors {name:(q,s,z,bits,g)}"""
    packed={}
    m=m.to(DEV)
    for l in range(len(m.bert.encoder.layer)):
        mods=encoder_linears(m,l); Hs={n:torch.zeros(mod.weight.shape[1],mod.weight.shape[1],device=DEV) for n,mod in mods.items()}
        cur={}
        hs=[]
        for n,mod in mods.items():
            def hook(mod_,inp,n=n):
                x=inp[0].float(); mk=cur["m"]; x=x[mk.bool()]
                Hs[n].add_(x.T@x)
            hs.append(mod.register_forward_pre_hook(hook))
        cnt=0
        for ids,tt,am in CB:
            cur["m"]=am.to(DEV); cnt+=int(am.sum())
            m(input_ids=ids.to(DEV),token_type_ids=tt.to(DEV),attention_mask=am.to(DEV))
        for h in hs: h.remove()
        for n,mod in mods.items():
            b=bitsfn(l,n)
            if b>=16: continue
            H=(2*Hs[n]/cnt).cpu()
            gl=g(l,n) if callable(g) else g
            q,s,z=gptq(mod.weight.data.cpu(),H,b,gl,damp=damp,clip=clip)
            mod.weight.data=dequant(q,s,z,gl).to(DEV)
            packed[f"L{l}.{n}"]=(q,s,z,b,gl)
        print(" layer",l,"done",flush=True)
    w=m.bert.embeddings.word_embeddings.weight.data.cpu()
    if embrest is None:
        q,s,z=quant_rtn(w,embbits,gemb); m.bert.embeddings.word_embeddings.weight.data=dequant(q,s,z,gemb).to(DEV); packed["emb"]=(q,s,z,embbits,gemb)
    else:   # full vocab: rows in embrest[0] (bool mask 'train-seen') get embbits, the rest embrest[1] bits
        mask=embrest[0]; out=torch.zeros_like(w)
        for nm,sel,bb in [("emb_a",mask,embbits),("emb_b",~mask,embrest[1])]:
            q,s,z=quant_rtn(w[sel],bb,gemb); out[sel]=dequant(q,s,z,gemb); packed[nm]=(q,s,z,bb,gemb)
        m.bert.embeddings.word_embeddings.weight.data=out.to(DEV)
    w=m.bert.pooler.dense.weight.data.cpu()
    q,s,z=quant_rtn(w,poolbits,64); m.bert.pooler.dense.weight.data=dequant(q,s,z,64).to(DEV); packed["pool"]=(q,s,z,poolbits,64)
    return m.cpu(),packed
def rep2(m,name):
    p=logits(m,B).argmax(-1).numpy(); a=p==y; ag=p==BP
    return f"{name}: devacc={a.mean():.4f} agree={ag.mean():.4f} | heldout(dev[1000:]) acc={a[NC:].mean():.4f} agree={ag[NC:].mean():.4f}"
def bytes_est(packed,m_extra=0):
    t=0
    for k,(q,s,z,b,g) in packed.items(): t+=q.numel()*b/8+s.numel()*4
    return t
if __name__=="__main__" and len(sys.argv)<2:
    import time
    for b,g in [(3,128),(3,64),(2,64),(2,32)]:
        t0=time.time()
        m,pk=gptq_quantize(fresh(),lambda l,n:b,g)
        print(rep2(m,f"GPTQ enc{b} g{g} emb4"),f"quantbytes~{bytes_est(pk)/1e6:.1f}MB",f"{time.time()-t0:.0f}s",flush=True)

# ---------- FFN neuron pruning (no training): score = mean|act| * ||W_out col||
@torch.no_grad()
def ffn_scores(m):
    m=m.to(DEV); sc=[torch.zeros(3072,device=DEV) for _ in m.bert.encoder.layer]; cur={}; n=[0]
    hs=[]
    for l,layer in enumerate(m.bert.encoder.layer):
        def hook(mod,inp,out,l=l):
            a=out.float().abs()*cur["m"].unsqueeze(-1); sc[l]+=a.sum((0,1))
        hs.append(layer.intermediate.register_forward_hook(hook))
    cnt=0
    for ids,tt,am in CB:
        cur["m"]=am.to(DEV).float(); cnt+=int(am.sum())
        m(input_ids=ids.to(DEV),token_type_ids=tt.to(DEV),attention_mask=am.to(DEV))
    for h in hs: h.remove()
    return [ (s/cnt).cpu()*layer.output.dense.weight.data.cpu().norm(dim=0) for s,layer in zip(sc,m.bert.encoder.layer)]
def prune_ffn(m,keepfrac):
    sc=ffn_scores(m); m=m.cpu()
    for l,layer in enumerate(m.bert.encoder.layer):
        k=int(round(3072*keepfrac/16))*16 if keepfrac<1 else 3072
        idx=sc[l].topk(k).indices.sort().values
        W1=layer.intermediate.dense; W2=layer.output.dense
        n1=torch.nn.Linear(768,k); n1.weight.data=W1.weight.data[idx].clone(); n1.bias.data=W1.bias.data[idx].clone()
        n2=torch.nn.Linear(k,768); n2.weight.data=W2.weight.data[:,idx].clone(); n2.bias.data=W2.bias.data.clone()
        layer.intermediate.dense=n1; layer.output.dense=n2
    m.config.intermediate_size=k
    return m
