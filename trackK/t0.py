import time;from embq import *;from ev import *
t=time.time()
cnt=train_counts();print('types seen',(cnt>0).sum(), 'tokens',cnt.sum(), time.time()-t)
r=dict(tiers=[dict(n=30522,bits=8,g=128)],pos=512)
p,l=build_embedding(r);print(len(p),time.time()-t)
print(evaluate(l),time.time()-t)
