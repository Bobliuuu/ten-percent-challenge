import sys; sys.argv=[sys.argv[0],"x"]
sys.path.insert(0,'/Users/jhome/ten-percent/trackA')
from gq import *
import time
names=dict(q="attention.self.query",k="attention.self.key",v="attention.self.value",a="attention.output.dense",i="intermediate.dense",o="output.dense")
inv={v:k for k,v in names.items()}
cfgs={"A q2k2v4a3i4o3":dict(q=2,k=2,v=4,a=3,i=4,o=3),
      "C q2k2v3a3i4o3":dict(q=2,k=2,v=3,a=3,i=4,o=3),
      "D q3k3v3a3i4o3":dict(q=3,k=3,v=3,a=3,i=4,o=3),
      "E q2k2v4a3i4o4":dict(q=2,k=2,v=4,a=3,i=4,o=4)}
for nm,c in cfgs.items():
    t0=time.time(); m,pk=gptq_quantize(fresh(),lambda l,n:c[inv[n]],128,embbits=3,clip=True)
    print(rep2(m,f"GPTQclip {nm} g128 emb3"),f"quantbytes~{bytes_est(pk)/1e6:.1f}MB",f"{time.time()-t0:.0f}s",flush=True)
