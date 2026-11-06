import sys,json,time,numpy as np;from embq import *;from ev import *
cnt=train_counts(); print('types seen',int((cnt>0).sum()),flush=True)
def r4(n,b,g=128): return dict(n=n,bits=b,g=g)
def lm(n,b,g=128): return dict(n=n,bits=b,g=g,lm=True)
N=30522
cfgs={
 'rtn8_all':[r4(N,8)],'rtn4_all':[r4(N,4)],'rtn3_all':[r4(N,3)],'rtn2_all':[r4(N,2)],
 'rtn4_g64':[r4(N,4,64)],'rtn3_g64':[r4(N,3,64)],'rtn2_g64':[r4(N,2,64)],'rtn2_g32':[r4(N,2,32)],
 'pq_ds16_k256':[dict(n=N,pq=dict(ds=16,k=256))],'pq_ds8_k256':[dict(n=N,pq=dict(ds=8,k=256))],'pq_ds4_k256':[dict(n=N,pq=dict(ds=4,k=256))],
 'pq_ds8_k64':[dict(n=N,pq=dict(ds=8,k=64))],'pq_ds12_k256':[dict(n=N,pq=dict(ds=12,k=256))],
 'top4k_4':[r4(4000,4)],'top8k_4':[r4(8000,4)],'top11k_4':[r4(11000,4)],
 'top8k_4+rest2':[r4(8000,4),r4(N-8000,2)],'top8k_3+rest2':[r4(8000,3),r4(N-8000,2)],'top6k_4+rest2':[r4(6000,4),r4(N-6000,2)],
 'top4k_4+rest2':[r4(4000,4),r4(N-4000,2)],'top4k_4+8k3+rest2':[r4(4000,4),r4(8000,3),r4(N-12000,2)],
 'top4k_4+rest_pq16':[r4(4000,4),dict(n=N-4000,pq=dict(ds=16,k=256))],
 'top4k_3+rest_pq16':[r4(4000,3),dict(n=N-4000,pq=dict(ds=16,k=256))],
 'top2k_4+rest_pq8':[r4(2000,4),dict(n=N-2000,pq=dict(ds=8,k=256))],
 'top2k_4+8k_pq8+rest_pq16':[r4(2000,4),dict(n=8000,pq=dict(ds=8,k=256)),dict(n=N-10000,pq=dict(ds=16,k=256))],
 'top1k_4+rest_pq8':[r4(1000,4),dict(n=N-1000,pq=dict(ds=8,k=256))],
}
cfgs.update({
 'lm2_all':[lm(N,2)],'lm2_g64':[lm(N,2,64)],'lm2_g256':[lm(N,2,256)],'lm3_all':[lm(N,3)],'lm3_g256':[lm(N,3,256)],'lm4_all':[lm(N,4)],'lm1_g64':[lm(N,1,64)],
 'top4k_lm4+rest_lm2':[lm(4000,4),lm(N-4000,2)],'top8k_lm3+rest_lm2':[lm(8000,3),lm(N-8000,2)],'top2k_lm4+rest_lm2':[lm(2000,4),lm(N-2000,2)],
 'top4k_lm4+rest_lm1':[lm(4000,4),lm(N-4000,1)],'top4k_lm3+rest_lm1_g64':[lm(4000,3),lm(N-4000,1,64)],
 'top8k_lm3+rest_pq8':[lm(8000,3),dict(n=N-8000,pq=dict(ds=8,k=256))],
 'top4k_lm4+8k_lm2+rest_pq8':[lm(4000,4),lm(8000,2),dict(n=N-12000,pq=dict(ds=8,k=256))],
})
cfgs['R4']=[lm(4000,4),lm(8000,2),lm(N-12000,1)]
cfgs['R5']=[lm(2000,4),lm(6000,3),lm(N-8000,1)]
import os
POS=int(os.environ.get('POS',512)); PB=int(os.environ.get('PB',0)) or None
sel=sys.argv[1:] 
for name,t in cfgs.items():
    if sel and not any(name==s for s in sel): continue
    for fb in (['nn','unk'] if name.startswith('top') and 'rest' not in name else ['nn']):
        t0=time.time(); p,l=build_embedding(dict(tiers=t,fallback=fb,pos=POS,pos_bits=PB))
        pb=len(p)-(0)  # exclude fp16 pos table to compare word-emb only; pos handled separately
        r=evaluate(l); r.update(name=name,pos=POS,pb=PB,fb=fb,bytes=len(p),word_bytes=pb,ent=l.ent_bytes,t=time.time()-t0)
        print(json.dumps(r),flush=True)
