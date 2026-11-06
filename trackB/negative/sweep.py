import numpy as np, itertools, run
from run import *
C=np.load("cache.npz",allow_pickle=True)
for name,sl_ in [("pooler",slice(0,768)),("cls",slice(768,1536)),("mean",slice(1536,2304))]:
    run.de,run.fe,run.se=[C[k][:,sl_] for k in ["dev_e","feedback_e","syn_e"]]
    mu=run.de.mean(0)
    for tau,k,al in itertools.product([0.5,0.6,0.7,0.8],[3,5],[3.0,8.0]):
        rs=[scenario(seed=s,mu=mu,tau=tau,k=k,alpha=al)[0][-1] for s in range(3)]
        a={x:np.mean([r[x] for r in rs]) for x in rs[0] if x!="step"}
        syn=np.mean([a[f"syn_b{i}"] for i in range(4)])
        print(name,tau,k,al,"dev %.3f syn %.3f errH %.3f bytes %d"%(a["dev"],syn,a["err_heldout"],a["bytes"]),flush=True)
