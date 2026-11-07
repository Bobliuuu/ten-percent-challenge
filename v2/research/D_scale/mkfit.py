import json,os,sys,numpy as np
sys.path.insert(0,os.path.expanduser("~/ten-percent/v2/harness")); from scorecard import retention
R="results/"
def pts(series):
    out=[]
    for b,c,m,ratio in series:
        if not os.path.exists(R+c): continue
        B=json.load(open(R+b))["t1"]; C=json.load(open(R+c))["t1"]; bc=np.array([r["correct"] for r in B]); cc=np.array([r["correct"] for r in C]); src=np.array([r["source"] for r in B]); kd=np.array([r["kind"] for r in B])
        o=retention(bc,cc)["ret"]; cells=[retention(bc,cc,(src==s)&(kd==k))["ret"] for s in ("S1","S3") for k in ("single","multi","nocall")]
        r=json.load(open(R+m))["ratio"] if m and os.path.exists(R+m) else ratio
        out.append((r,o,min(cells)))
    return out
def fit(p,idx,target):
    x=np.log([q[0] for q in p]); y=np.log((1-np.clip([q[idx] for q in p],1e-3,.9995))/np.clip([q[idx] for q in p],1e-3,.9995)); A=np.polyfit(x,y,1)
    xt=(np.log((1-target)/target)-A[1])/A[0]; return A[0],float(np.exp(xt))
S={"32B plain alloc (legacy calib)":[("v_base32.json","v_t096.json","meta32_t096.json",0.0961),("v_base32.json","v_t12.json","meta32_t12.json",0.1203),("v_base32.json","v_t15.json","meta32_t15.json",0.1503)],
"32B gamma=1 (clean calib)":[("v_base32.json","v_C32_gam1.json",None,0.0962),("v_base32.json","v_C32_gam1_r12.json","meta32_r12g1.json",0.1203),("v_base32.json","v_C32_gam1_r15.json",None,0.1502)]}
for n,s in S.items():
    p=pts(s); print(f"* {n}: points (ratio, overall ret, worst cell): "+"; ".join(f"({a:.4f}, {b*100:.1f}%, {c*100:.1f}%)" for a,b,c in p))
    if len(p)>=3:
        sl,r1=fit(p,1,.98); sl2,r2=fit(p,2,.95); print(f"  logit(fail) vs ln(ratio) slope {sl:.2f}: overall >=98% extrapolated at **{r1*100:.1f}%** of fp16; worst-cell >=95% at **{r2*100:.1f}%** (3-point fit, no CI; treat as +-3 pts)")
    elif len(p)==2:
        sl,r1=fit(p,1,.98); print(f"  2-point slope {sl:.2f}: overall >=98% at ~{r1*100:.1f}% (two points only)")
print("* Scale axis at 9.6% (same recipe where possible): Qwen3-8B 21.9% and Qwen3-14B 56.1% (plain, legacy items, other agent) -> Qwen3-32B 74.6% (plain, clean) -> Qwen2.5-72B 73.9% (plain, legacy calib; different family). With gamma=1: 32B 83.8%, 72B 82.6%. Retention does not increase from 32B to 72B at 9.6%, so no log(params) extrapolation to 123B/405B is supported by this data; if the 8B->14B->32B trend (+35, +19 pts) continued it would predict ~90% at 70B+, but the 72B point shows the trend broke (family/base effects; the 72B base was fp8). For 405B the honest statement is: unknown, the only direct evidence says size alone does not cure 1.5 bpw.")
