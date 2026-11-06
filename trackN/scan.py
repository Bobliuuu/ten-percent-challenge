from power import *
import numpy as np
print("80% thresholds by scan of d (delta=0 symmetric)")
for d in (.001,.002,.003,.004,.005,.006,.008):
    o=probs(d/2,d/2); print(d, round(d*n,1),{k:round(v,3) for k,v in o.items()},flush=True)
print("delta=+0.2pt correcting:")
for d in (.005,.01,.02):
    dl=.002; o=probs((d-dl)/2,(d+dl)/2); print(d,round(o['ni'],3))
# sample size for NI at delta=0 to pass w/ 80%: approx scaling, use normal approx
from scipy.stats import norm
za,zb=norm.ppf(.95),norm.ppf(.8)
for d in (.01,.02,.05):
    # NI: need delta_true=0 vs margin .01: n=(za+zb)^2*d/.01^2
    print("NI n for d",d,round((za+zb)**2*d/1e-4), " detect 1pt drop two-sided .05:",round((1.96+zb)**2*d/1e-4))
p=.92; print("unpaired n/arm 1pt",round((1.96+zb)**2*2*p*(1-p)/1e-4))
