import sys; sys.argv=[sys.argv[0],"x"]
sys.path.insert(0,'/Users/jhome/ten-percent/trackA')
from gq import *
import json
base_m=fresh(); ref=logits(base_m,B)
res={}
for l in range(12):
    for n in LIN:
        m=fresh(); mod=encoder_linears(m,l)[n]
        q,s,z=quant_rtn(mod.weight.data,3,128); mod.weight.data=dequant(q,s,z,128)
        lg=logits(m,B); res[f"{l}.{n}"]=float((lg-ref).pow(2).mean())
    print(l,{n:round(res[f"{l}.{n}"],4) for n in LIN},flush=True)
json.dump(res,open("trackA/sens.json","w"))
