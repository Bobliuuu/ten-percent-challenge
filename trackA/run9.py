exec(open('/Users/jhome/ten-percent/trackA/run7.py').read().split("for bits,g in")[0])
import gq, time
ID=torch.arange(tok.vocab_size)
gq.CB=encode(tx[:1000],ID)
BF=encode(tx,ID)
mask=torch.zeros(tok.vocab_size,dtype=torch.bool); mask[torch.tensor(keep)]=True
names=dict(q="attention.self.query",k="attention.self.key",v="attention.self.value",a="attention.output.dense",i="intermediate.dense",o="output.dense")
inv={v:k for k,v in names.items()}
def mk(**kw):
    d=dict(q=3,k=3,v=3,a=3,i=3,o=3); d.update(kw)
    def f(l,n):
        x=d[inv[n]]; return x[l] if isinstance(x,list) else x
    return f
G256=lambda l,n: 256 if inv[n]=="o" else 128
def r4(m,name,pk):
    s=[name+f" quantbytes~{bytes_est(pk)/1e6:.1f}"]
    for nm,(b,l,bp) in Dfull.items():
        p=logits(m,b).argmax(-1).numpy(); s.append(f"{nm}: acc={np.mean(p==l):.4f} agree={np.mean(p==bp):.4f}")
    p=logits(m,BF).argmax(-1).numpy(); s.append(f"olddev acc={np.mean(p==y):.4f} agree={np.mean(p==BP):.4f}")
    print(" | ".join(s),flush=True)
cfgs=[("P uniform3 g128, emb 3/2",mk(),128,3,2),
      ("Q q2k2 i4(l<6) rest3 go256, emb 3/2",mk(q=2,k=2,i=[4]*6+[3]*6),G256,3,2),
      ("R uniform3 g128, emb 2/2",mk(),128,2,2)]
for nm,bf,gf,e1,e2 in cfgs:
    m,pk=gq.gptq_quantize(base_model(),bf,gf,embbits=e1,clip=True,embrest=(mask,e2)); r4(m.cpu(),"GPTQclip FULLVOCAB "+nm,pk)
