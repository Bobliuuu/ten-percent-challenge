import sys; sys.argv=[sys.argv[0],"x"]
sys.path.insert(0,'/Users/jhome/ten-percent/trackA'); sys.path.insert(0,'/Users/jhome/ten-percent/trackG')
from gq import *
from dev2 import load_dev2
D={}
for nm,h in [("dev2",False),("hard",True)]:
    t,l=load_dev2(hard=h); D[nm]=(encode(t,T),np.array(l),logits(base,encode(t,torch.arange(tok.vocab_size))).argmax(-1).numpy())
def r3(m,name):
    s=[name]
    for nm,(b,l,bp) in D.items():
        p=logits(m,b).argmax(-1).numpy(); s.append(f"{nm}: acc={np.mean(p==l):.4f} agree={np.mean(p==bp):.4f}")
    p=logits(m,B).argmax(-1).numpy(); s.append(f"olddev acc={np.mean(p==y):.4f} agree={np.mean(p==BP):.4f}")
    print(" | ".join(s),flush=True)
