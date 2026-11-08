"""All qlib methods as {name: fn(W,H)->(What,info)} for the per-expert layer-proxy table (unbatched; slow, used on a few experts only)."""
import qlib as Q

def methods():
    S = Q.ScalarQ; had = ("had", 1); M = {}
    for b in (2, 3):
        M[f"rtn_int{b}_g128"] = Q.m_rtn(lambda b=b: S("int", b, 128, clip=True))
        M[f"rtn_int{b}_g128_had"] = Q.m_rtn(lambda b=b: S("int", b, 128, clip=True), rot=had)
        M[f"gptq_int{b}_g128"] = Q.m_gptq(lambda b=b: S("int", b, 128))
        M[f"gptq_int{b}_g128_had"] = Q.m_gptq(lambda b=b: S("int", b, 128), rot=had)
        M[f"gptq_int{b}_g128_clip_had"] = Q.m_gptq(lambda b=b: S("int", b, 128, clip=True), rot=had)
        M[f"gptq_int{b}_g128_actorder_had"] = Q.m_gptq(lambda b=b: S("int", b, 128), rot=had, act_order=True)
        M[f"gptq_int{b}_g32_had"] = Q.m_gptq(lambda b=b: S("int", b, 32), rot=had)
        M[f"gptq_int{b}_g64_had"] = Q.m_gptq(lambda b=b: S("int", b, 64), rot=had)
        M[f"gptq_int{b}_g256_had"] = Q.m_gptq(lambda b=b: S("int", b, 256), rot=had)
        M[f"nf{b}_g64_gptq_had"] = Q.m_gptq(lambda b=b: S("grid", levels=Q.nf_levels(b), g=64, clip=True), rot=had)
        M[f"hqq_int{b}_g64"] = Q.m_hqq(b, 64)
        M[f"awq_int{b}_g128"] = Q.m_awq(Q.m_rtn(lambda b=b: S("int", b, 128, clip=True)))
        M[f"awq_gptq_int{b}_g128"] = Q.m_awq(Q.m_gptq(lambda b=b: S("int", b, 128)))
        M[f"smoothquant+_int{b}_g128"] = Q.m_smoothquant_plus(Q.m_rtn(lambda b=b: S("int", b, 128, clip=True)))
        M[f"omniquant_lwc_search_int{b}_g128"] = Q.m_omniquant_lwc(b, 128, steps=0)
        M[f"omniquant_lwc_LEARNED_int{b}_g128"] = Q.m_omniquant_lwc(b, 128, steps=40)
        M[f"owq1pct_int{b}_g128"] = Q.m_owq(lambda b=b: S("int", b, 128), 0.01)
        M[f"spqr1pct_int{b}_g16"] = Q.m_spqr(lambda b=b: S("int", b, 16), target_frac=0.01)
        M[f"squeezellm_int{b}"] = Q.m_squeezellm(b)
        M[f"admmq_int{b}_g128"] = Q.m_admmq(b, 128)
        M[f"lqer_r16_int{b}_g128_gptq_had(GRAY adapter)"] = Q.m_lowrank(Q.m_gptq(lambda b=b: S("int", b, 128), rot=had), rank=16, iters=1)
        M[f"caldera_r16_it3_int{b}_g128_gptq_had(GRAY adapter)"] = Q.m_lowrank(Q.m_gptq(lambda b=b: S("int", b, 128), rot=had), rank=16, iters=3)
    M["fp4_e2m1_mxfp4_g32_gptq"] = Q.m_gptq(lambda: S("grid", levels=Q.FP4, g=32, scale_fmt="e8m0", clip=True))
    M["fp4_nvfp4_g16_gptq"] = Q.m_gptq(lambda: S("grid", levels=Q.FP4, g=16, scale_fmt="e4m3", clip=True))
    M["nf4_g64_gptq"] = Q.m_gptq(lambda: S("grid", levels=Q.nf_levels(4), g=64, clip=True))
    M["int4_g128_gptq_had"] = Q.m_gptq(lambda: S("int", 4, 128), rot=had)
    for t in (3.0, 2.0, 1.5, 1.0):
        M[f"ecsq_{t}_g256_had"] = Q.m_gptq(lambda t=t: S("ecsq", g=256, target_bpw=t), rot=had)
        M[f"ecsq_rdo_{t}_g256_had"] = Q.m_gptq(lambda t=t: S("ecsq", g=256, target_bpw=t, lam=0.05), rot=had)
        M[f"ecsq_{t}_g64_had"] = Q.m_gptq(lambda t=t: S("ecsq", g=64, target_bpw=t), rot=had)
        M[f"waterfill_ecsq_{t}_perrow_had"] = Q.m_gptq(lambda t=t: S("ecsq", g=None, target_bpw=t), rot=had)
    for b in (3.0, 2.0, 1.5, 1.0):
        D = 4; M[f"vq_d4_{b}_had"] = Q.m_gptq(lambda b=b: Q.VQ(4, int(round(2 ** (b * 4)))), rot=had)
        M[f"vq_d8_{b}_had"] = Q.m_gptq(lambda b=b: Q.VQ(8, int(round(2 ** (b * 8)))), rot=had) if b <= 1.5 else None
        M[f"rvq_aqlm_style_d8_2x_{b}_had"] = Q.m_gptq(lambda b=b: Q.VQ(8, [int(round(2 ** (b * 4)))] * 2, stages=2, fit=False), rot=had)
        M[f"rvq_fitted_d8_2x_{b}_had"] = Q.m_gptq(lambda b=b: Q.VQ(8, [int(round(2 ** (b * 4)))] * 2, stages=2, fit=True), rot=had)
        M[f"gptvq_d2_{b}"] = Q.m_gptq(lambda b=b: Q.GPTVQ(2, b, 256))
        M[f"gptvq_d4_{b}_had"] = Q.m_gptq(lambda b=b: Q.GPTVQ(4, b, 256), rot=had)
    M["e8_lattice_1.5_had"] = Q.m_gptq(lambda: Q.VQ(8, 4096, "e8"), rot=had)
    M["e8_lattice_1.375_had"] = Q.m_gptq(lambda: Q.VQ(8, 2048, "e8"), rot=had)
    M["tcq_L12_V2_kb2_had"] = Q.m_gptq(lambda: Q.TCQ(12, 2, 2), rot=had)
    M["tcq_L12_V2_kb3_had"] = Q.m_gptq(lambda: Q.TCQ(12, 2, 3), rot=had)
    M["tcq_L12_V2_kb4_had"] = Q.m_gptq(lambda: Q.TCQ(12, 2, 4), rot=had)
    M["sparsegpt_2:4_int2_g128"] = Q.m_sparse_quant(lambda: S("int", 2, 128), "sparsegpt", nm=(2, 4))
    M["sparsegpt_50_int3_g128"] = Q.m_sparse_quant(lambda: S("int", 3, 128), "sparsegpt", sp=0.5)
    M["wanda_50_int3_g128"] = Q.m_sparse_quant(lambda: S("int", 3, 128), "wanda", sp=0.5)
    M["wanda_2:4_int2_g128"] = Q.m_sparse_quant(lambda: S("int", 2, 128), "wanda", nm=(2, 4))
    return {k: v for k, v in M.items() if v is not None}
