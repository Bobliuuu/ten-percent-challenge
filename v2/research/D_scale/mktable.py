import json,os,sys,numpy as np
sys.path.insert(0,os.path.expanduser("~/ten-percent/v2/harness")); from scorecard import retention
R="results/"
def L(f): return json.load(open(R+f))["t1"]
rows=[
("Qwen3-32B plain alloc (gamma0), LEGACY calib","v_base32.json","v_t096.json","meta32_t096.json","legacy"),
("Qwen3-32B plain alloc 12.0%, LEGACY calib","v_base32.json","v_t12.json","meta32_t12.json","legacy"),
("Qwen3-32B plain alloc 15.0%, LEGACY calib","v_base32.json","v_t15.json","meta32_t15.json","legacy"),
("Qwen3-32B gamma0, CLEAN calib","v_base32.json","v_C32_gam0.json",None,"clean"),
("Qwen3-32B gamma1, CLEAN calib","v_base32.json","v_C32_gam1.json",None,"clean"),
("Qwen3-32B R3 measured-KL + gamma1, CLEAN","v_base32.json","v_C32_R3g1.json","meta32_C32R3g1.json","clean"),
("Qwen3-32B gamma1 nocall-heavy calib, CLEAN","v_base32.json","v_C32_gam1_nc.json",None,"clean"),
("Qwen3-32B gamma1 12.0%, CLEAN","v_base32.json","v_C32_gam1_r12.json",None,"clean"),
("Qwen3-32B gamma1 15.0%, CLEAN","v_base32.json","v_C32_gam1_r15.json",None,"clean"),
("Qwen3-32B gamma1 + SELF-DISTILLED calib, CLEAN","v_base32.json","v_C32_gam1_self.json",None,"clean"),
("Qwen3-32B gamma1, LEGACY calib","v_base32.json","v_L2_gam1.json","meta32_gam1.json","legacy"),
("Qwen3-32B gamma2, LEGACY calib","v_base32.json","v_L4_gam2.json","meta32_gam2.json","legacy"),
("Qwen3-32B R3 measured-KL (no gamma), LEGACY","v_base32.json","v_R3_kl.json","meta32_R3.json","legacy"),
("Qwen3-32B ncal256, LEGACY","v_base32.json","v_L1_ncal256.json","meta32_L1.json","legacy"),
("Qwen3-32B label+tool calib mix, LEGACY tool part","v_base32.json","v_LCM_mix.json",None,"legacy"),
("Qwen3-32B gamma1 + label mix, LEGACY","v_base32.json","v_L5_gam1_lcm.json",None,"legacy"),
("Qwen2.5-72B plain alloc, LEGACY calib","base72_fp8dyn.json","v_72_t096.json","meta72_t096.json","legacy"),
("Qwen2.5-72B gamma1, CLEAN calib","base72_fp8dyn.json","v_72_C72_gam1.json","meta72_C72g1.json","clean"),
("Qwen2.5-72B V3 gamma1+ncal256+emb8+label-mix calib, CLEAN","base72_fp8dyn.json","v_72_V3_gam1_ncal256_emb8_lcm.json",None,"clean"),
("Qwen2.5-72B V7 V3 + self-distilled tool part, CLEAN","base72_fp8dyn.json","v_72_V7_gam1_ncal256_emb8_lcm_self.json",None,"clean"),
("Qwen2.5-72B gamma1+ncal256+emb/head 8 lvls, CLEAN","base72_fp8dyn.json","v_72_V12_gam1_ncal256_emb8.json",None,"clean"),
]
print("| model / recipe | calib | ratio | retention (Wilson lo) | S1 | S3 | worst cell | strict |\n|---|---|---|---|---|---|---|---|")
for n,b,c,m,cal in rows:
    if not os.path.exists(R+c): continue
    B=L(b); C=L(c); bc=np.array([r["correct"] for r in B]); cc=np.array([r["correct"] for r in C]); src=np.array([r["source"] for r in B]); kd=np.array([r["kind"] for r in B])
    o=retention(bc,cc); s1=retention(bc,cc,src=="S1")["ret"]; s3=retention(bc,cc,src=="S3")["ret"]
    cells={f"{s}/{k}":retention(bc,cc,(src==s)&(kd==k))["ret"] for s in ("S1","S3") for k in ("single","multi","nocall")}; w=min(cells,key=cells.get)
    ok=o["ret"]>=.98 and o["ret_lo"]>=.98 and min(cells.values())>=.95
    ratio=json.load(open(R+m))["ratio"] if m and os.path.exists(R+m) else None
    print(f"| {n} | {cal} | {('%.4f'%ratio) if ratio else 'see note'} | {o['ret']*100:.1f}% ({o['ret_lo']*100:.1f}) | {s1*100:.1f} | {s3*100:.1f} | {w} {cells[w]*100:.1f} | {'PASS' if ok else 'fail'} |")
