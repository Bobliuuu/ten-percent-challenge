import sys,os
LAYER=sys.argv[1]; sys.argv=['x','feats/f2','q15',LAYER]
exec(open('ana.py').read().split('if __name__')[0])
for tn in TASKS:
    pr=dict(w=3,lam=.5,eta=.5,ba=1,tau=.9,r=96); b,o=scenB(tn,pr,'pool')
    print(tn,'base acc',round(b['acc'],3),{k:{m:(round(o[k][m]['fix'],2),round(o[k][m]['acc'],3)) for m in o[k]} for k in (5,25)})
