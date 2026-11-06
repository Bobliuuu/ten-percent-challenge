import sys; sys.argv=[sys.argv[0],"x"]
sys.path.insert(0,'/Users/jhome/ten-percent/trackA')
from gq import *
import time
names=dict(q="attention.self.query",k="attention.self.key",v="attention.self.value",a="attention.output.dense",i="intermediate.dense",o="output.dense")
inv={v:k for k,v in names.items()}
def mk(q=2,k=2,v=4,a=3,i=4,o=(4,)*6+(3,)*6):
    def f(l,n):
        t=inv[n]; x=dict(q=q,k=k,v=v,a=a,i=i,o=o)[t]
        return x[l] if isinstance(x,tuple) else x
    return f
G256=lambda l,n: 256 if inv[n]=="o" else 128
cfgs={"F q2k2v4a3i4 o4(l<6)/3 go256":(mk(),G256),
      "G q2k2v4a3i4 o4(l<8)/3 go256 (o2 l>=10)":(mk(o=(4,)*8+(3,)*2+(2,)*2),G256),
      "H q2k2v3a3i4 o4(l<8)/3 go256":(mk(v=3,o=(4,)*8+(3,)*4),G256),
      "I q2k2v4a3 i4(l<10)/3 o4(l<8)/3 go256":(mk(i=(4,)*10+(3,)*2,o=(4,)*8+(3,)*4),G256)}
for nm,(bf,gf) in cfgs.items():
    t0=time.time(); m,pk=gptq_quantize(fresh(),bf,gf,embbits=3,clip=True)
    print(rep2(m,f"GPTQclip {nm} emb3"),f"quantbytes~{bytes_est(pk)/1e6:.1f}MB",f"{time.time()-t0:.0f}s",flush=True)
