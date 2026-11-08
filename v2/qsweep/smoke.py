import torch, qlib as Q, time
torch.manual_seed(0); dev="cpu"
R,C=256,512
W=torch.randn(R,C)*0.02; X=torch.randn(4000,C)*(1+3*torch.rand(C)); H=X.T@X
ms={
 "rtn_int2_g64":Q.m_rtn(lambda:Q.ScalarQ("int",2,64)),
 "gptq_int2_g64":Q.m_gptq(lambda:Q.ScalarQ("int",2,64)),
 "gptq_ecsq1.5_rot":Q.m_gptq(lambda:Q.ScalarQ("ecsq",g=128,target_bpw=1.5),rot=("had",1)),
 "gptq_ecsq1.5_rdo":Q.m_gptq(lambda:Q.ScalarQ("ecsq",g=128,target_bpw=1.5,lam=0.05),rot=("had",1)),
 "gptq_vq8_k4096_rot":Q.m_gptq(lambda:Q.VQ(8,4096),rot=("had",1)),
 "gptq_e8_rot":Q.m_gptq(lambda:Q.VQ(8,4096,"e8"),rot=("had",1)),
 "rvq2x64":Q.m_gptq(lambda:Q.VQ(8,[64,64],stages=2),rot=("had",1)),
 "gptvq_d2":Q.m_gptq(lambda:Q.GPTVQ(2,1.5,128)),
 "nf2_g64":Q.m_gptq(lambda:Q.ScalarQ("grid",levels=Q.nf_levels(2),g=64)),
 "fp4":Q.m_gptq(lambda:Q.ScalarQ("grid",levels=Q.FP4,g=32,scale_fmt="e8m0")),
 "hqq2":Q.m_hqq(2,64),"squeeze2":Q.m_squeezellm(2),"admm2":Q.m_admmq(2,64),
 "awq":Q.m_awq(Q.m_rtn(lambda:Q.ScalarQ("int",2,64))),
 "omni":Q.m_omniquant_lwc(2,64),"owq":Q.m_owq(lambda:Q.ScalarQ("int",2,64)),
 "spqr":Q.m_spqr(lambda:Q.ScalarQ("int",2,16)),
 "sgpt24_int2":Q.m_sparse_quant(lambda:Q.ScalarQ("int",2,64),"sparsegpt",nm=(2,4)),
 "wanda50":Q.m_sparse_quant(lambda:Q.ScalarQ("int",2,64),"wanda",0.5),
 "lowrank":Q.m_lowrank(Q.m_gptq(lambda:Q.ScalarQ("int",2,64)),rank=16,iters=2),
}
for k,f in ms.items():
    t=time.time()
    try:
        Wh,i=f(W,H); print(f"{k:22s} relerr {Q.relerr(W,Wh,H):.4f} bpw {i['bits']/W.numel():.3f} {time.time()-t:.1f}s")
    except Exception as e:
        import traceback; print(k,"FAIL",repr(e)[:200])
