import sys, numpy as np, functools
sys.path.insert(0,'../trackC'); import common as C
from scipy.stats import multinomial, binomtest
n=872
@functools.lru_cache(None)
def ni(b,c): return C.ni_pvalue(n,b,c,-0.01)<0.05
@functools.lru_cache(None)
def mc(b,c): return C.mcnemar_exact(b,c)>0.05
def probs(pb,pc,K=None):
    # exact enumeration over (b,c)
    from scipy.stats import binom
    out={'ni':0,'eq':0,'w1':0,'w05':0,'mc':0}
    for N in range(0,min(n,120)+1):
        pN=binom.pmf(N,n,pb+pc)
        if pN<1e-12: continue
        for c in range(N+1):
            b=N-c
            w=pN*binom.pmf(c,N,pc/(pb+pc))
            if ni(b,c): out['ni']+=w
            if b==c: out['eq']+=w
            if abs(b-c)<=1: out['w1']+=w
            if abs(b-c)<=0.005*n: out['w05']+=w
            if mc(b,c): out['mc']+=w
    return out
if __name__=='__main__':
    print("base,delta,d,feasible,NI,eq,w1,w05,mcnemar")
    for acc in (.90,.92,.93,.94):
      for dl in (0,-.003,-.005,-.01):
        for d in (.005,.01,.02,.03,.05):
            pc=(d+dl)/2; pb=(d-dl)/2
            feas = pc<=1-acc and pb<=acc and pc>=0
            if not feas: print(f"{acc},{dl},{d},NO"); continue
            o=probs(pb,pc)
            print(f"{acc},{dl},{d},yes,"+",".join(f"{o[k]:.3f}" for k in ('ni','eq','w1','w05','mc')),flush=True)
