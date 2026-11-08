import torch, qlib as Q, qlib_b as B
torch.manual_seed(0); E,R,C=4,128,256
W=torch.randn(E,R,C)*0.02; X=torch.randn(E,3000,C)*(1+3*torch.rand(E,1,C)); H=X.transpose(1,2)@X
for name,fn,fnu in [("int2g64",B.mb_gptq(lambda:B.IntQ(2,64)),Q.m_gptq(lambda:Q.ScalarQ("int",2,64))),
                    ("int2g64rot",B.mb_gptq(lambda:B.IntQ(2,64),rot=("had",1)),Q.m_gptq(lambda:Q.ScalarQ("int",2,64),rot=("had",1))),
                    ("vq4",B.mb_gptq(lambda:B.VQ(4,256),rot=("had",1)),Q.m_gptq(lambda:Q.VQ(4,256),rot=("had",1))),
                    ("ecsq",B.mb_gptq(lambda:B.ECSQ(2.0,64),rot=("had",1)),Q.m_gptq(lambda:Q.ScalarQ("ecsq",g=64,target_bpw=2.0),rot=("had",1)))]:
    Wh,b=fn(W,H); e=B.relerr_b(W,Wh,H)
    eu=[Q.relerr(W[i],fnu(W[i],H[i])[0],H[i]) for i in range(E)]
    print(name,"batched",[round(float(x),4) for x in e],"bpw",float(b.mean())/(R*C),"| single",[round(x,4) for x in eu])
