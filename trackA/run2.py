import sys; sys.argv=[sys.argv[0],"x"]+sys.argv[1:]
sys.path.insert(0,'/Users/jhome/ten-percent/trackA')
from gq import *
import time
for b,g,e in [(3,128,3),(3,128,2)]:
    t0=time.time(); m,pk=gptq_quantize(fresh(),lambda l,n:b,g,embbits=e,clip=True)
    print(rep2(m,f"GPTQclip enc{b} g{g} emb{e}"),f"quantbytes~{bytes_est(pk)/1e6:.1f}MB",f"{time.time()-t0:.0f}s",flush=True)
