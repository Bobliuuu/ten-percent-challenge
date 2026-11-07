import json,os,math,collections
R="results/"
def wil(k,n,z=1.96):
    if n==0: return 0.0
    p=k/n;d=1+z*z/n;c=p+z*z/(2*n);h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n));return (c-h)/d
def ret(b,c):
    B={r["id"]:r for r in json.load(open(R+b))["rows"]}; C={r["id"]:r for r in json.load(open(R+c))["rows"]}; out={}
    for t in ["INJ","TOOLSEL","SENT","FIN","ROUTE","TOX","NLI","SPAM","AIDET"]:
        g=collections.defaultdict(lambda:[0,0])
        for i,x in B.items():
            if x["task"]==t and x["correct"]:
                for key in ("ALL","r:"+x["role"],"k:"+x["kind"]): g[key][1]+=1; g[key][0]+=C[i]["correct"]
        a=g["ALL"]; cells=[v[0]/v[1] for k,v in g.items() if k!="ALL" and v[1]>=10]
        out[t]=(a[0]/a[1],wil(*a),min(cells))
    return out
arts=[("32B 9.6% plain, legacy","ct_Qwen3-32B_bf16.json","ct_Qwen3-32B_t096.json"),("32B 12.0% plain, legacy","ct_Qwen3-32B_bf16.json","ct_Qwen3-32B_t12.json"),("32B 15.0% plain, legacy","ct_Qwen3-32B_bf16.json","ct_Qwen3-32B_t15.json"),
("32B 9.6% gamma1, legacy","ct_Qwen3-32B_bf16.json","ct_Qwen3-32B_L2_gam1.json"),("32B 9.6% R3-KL, legacy","ct_Qwen3-32B_bf16.json","ct_Qwen3-32B_R3_kl.json"),("32B 9.6% label-mix calib (LCM)","ct_Qwen3-32B_bf16.json","ct_Qwen3-32B_LCM_mix.json"),
("32B 12.0% gamma1, clean","ct_Qwen3-32B_bf16.json","ct_Qwen3-32B_C32_gam1_r12.json"),("72B 9.6% plain, legacy","ct_Qwen2.5-72B_bf16.json","ct_Qwen2.5-72B_t096.json"),("72B 9.6% V3 gamma1+ncal256+emb8+label-mix, clean","ct_Qwen2.5-72B_bf16.json","ct_Qwen2.5-72B_V3_gam1_ncal256_emb8_lcm.json"),("72B 9.6% V6 same, pure label calib","ct_Qwen2.5-72B_bf16.json","ct_Qwen2.5-72B_V6_gam1_ncal192_emb8_lblonly.json"),("72B 9.6% V7 label-mix + self-distilled tool part","ct_Qwen2.5-72B_bf16.json","ct_Qwen2.5-72B_V7_gam1_ncal256_emb8_lcm_self.json"),("32B 15.0% gamma1, clean","ct_Qwen3-32B_bf16.json","ct_Qwen3-32B_C32_gam1_r15.json"),("72B 9.6% gamma1, clean","ct_Qwen2.5-72B_bf16.json","ct_Qwen2.5-72B_C72_gam1.json"),("72B 9.6% gamma1+ncal256+emb8, clean","ct_Qwen2.5-72B_bf16.json","ct_Qwen2.5-72B_V12_gam1_ncal256_emb8.json")]
tasks=["INJ","TOOLSEL","SENT","FIN","ROUTE","TOX","NLI","SPAM","AIDET"]
print("| artifact | "+" | ".join(tasks)+" |\n|---|"+"---|"*len(tasks))
for n,b,c in arts:
    if not os.path.exists(R+c): continue
    r=ret(b,c); print(f"| {n} | "+" | ".join(f"{r[t][0]*100:.1f} ({r[t][1]*100:.1f}/{r[t][2]*100:.0f})" for t in tasks)+" |")
