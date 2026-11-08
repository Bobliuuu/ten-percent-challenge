"""Exact-ish byte budget for giant MoEs at 10% of fp16 (2 B/param).  Expert params from public configs.
nonexp = everything else (attn, shared experts, dense layers, router, embed+head, norms).  Non-expert bits are charged incl. fp16 g128 scale.
Output: bits/param left for routed experts (all-in, incl. scales/codebooks/entropy-coded) after non-expert at NB bits."""
# name: (total_params_B, routed_expert_params_B, note)
M = {
 "DeepSeek-V3/R1 (671B)":      (671.0, 58*256*3*7168*2048/1e9, "58 MoE layers x256 experts"),
 "Kimi-K2 (1.03T)":            (1026.0, 60*384*3*7168*2048/1e9, "60 MoE x384"),
 "Qwen3-235B-A22B":            (235.1, 94*128*3*4096*1536/1e9, "94 x128"),
 "Qwen3-Coder-480B-A35B":      (480.2, 62*160*3*6144*2560/1e9, "62 x160"),
 "GLM-4.5 (355B)":             (358.0, 89*160*3*5120*1536/1e9, "89 MoE x160"),
 "Llama-4-Maverick (400B)":    (401.6, 24*128*3*5120*8192/1e9, "24 MoE x128"),
 "Llama-4-Scout (109B)":       (108.6, 48*16*3*5120*8192/1e9, "48 x16"),
 "gpt-oss-120b":               (116.8, 36*128*3*2880*2880/1e9, "36 x128, native MXFP4 4.25b"),
}
print(f"{'model':26}{'fp16 GB':>9}{'10% GB':>8}{'exp share':>10}{'nonexp B':>9}{'exp bpw@NB=4.5':>15}{'@6':>6}{'@8':>6}")
for n,(tot,ex,_) in M.items():
    b=tot*2; bud=b*.1; ne=tot-ex
    f=lambda nb:(bud-0.016-ne*nb/8)*8/ex   # GB*8/Bparam = bits/param
    print(f"{n:26}{b:9.0f}{bud:8.1f}{ex/tot:10.3f}{ne:9.1f}{f(4.5):15.3f}{f(6):6.2f}{f(8):6.2f}")
