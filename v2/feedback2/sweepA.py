import sys,os
LAYER,R,AL=sys.argv[1:4]; os.environ['ALPHA']=AL; sys.argv=['x','feats/f2',sys.argv[4] if len(sys.argv)>4 else 'q15',LAYER]
exec(open('ana.py').read().split('if __name__')[0])
o_=[]
for tn in HOLD:
    pr=dict(n0=1,w=2,lam=.3,st=.2,sa=3,r=int(R)); res,o=scenA(tn,pr,'pool')
    o_.append(f"{tn[:4]} k5 {o[5]['lda_pure']['new']:.2f}/{o[5]['lda_pure']['old']:.2f} k25 {o[25]['lda_pure']['new']:.2f}/{o[25]['lda_pure']['old']:.2f} (b {res['base']['new']:.2f}/{res['base']['old']:.2f})")
print(f"L={LAYER} r={R} a={AL} | "+" | ".join(o_))
