exec(open('/Users/jhome/ten-percent/trackA/ev2.py').read())
ID=torch.arange(tok.vocab_size)
Dfull={}
for nm,h in [("dev2",False),("hard",True)]:
    t,l=load_dev2(hard=h); Dfull[nm]=(encode(t,ID),np.array(l),D[nm][2])
def r3f(m,name):
    s=[name]
    for nm,(b,l,bp) in Dfull.items():
        p=logits(m,b).argmax(-1).numpy(); s.append(f"{nm}: acc={np.mean(p==l):.4f} agree={np.mean(p==bp):.4f}")
    print(" | ".join(s),flush=True)
for bits,g in [(4,128),(3,128),(2,128),(2,64),(2,32)]:
    m=base_model(); w=m.bert.embeddings.word_embeddings.weight.data
    q,s,z=quant_rtn(w,bits,g); m.bert.embeddings.word_embeddings.weight.data=dequant(q,s,z,g)
    r3f(m,f"FULL vocab emb{bits} g{g} (enc fp32) bytes={w.numel()*bits/8/1e6+s.numel()*4/1e6:.1f}MB")
