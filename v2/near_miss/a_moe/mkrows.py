import json, glob, os, sys, re
sys.argv=["x"]
exec(open("ret.py").read().split("def main")[0])
import collections
def wl_(k,n): return round(wl(k,n),3)
rows=[]
for d,base,bname in (("b2","base_qwen15.json","Qwen1.5-MoE-A2.7B-Chat (14.3B)"),("dsv2l","base_dsv2l.json","DeepSeek-V2-Lite-Chat (15.7B)"),("olmoe","base_olmoe.json","OLMoE-1B-7B-0125-Instruct (6.9B)")):
    B=rows_=None
    Bm=globals()["rows"] if False else None
import json
def R(p): return {r["id"]: (ok(r["out"], r["gold"]), (r["src"], r["gold"])) for r in json.load(open(p))["rows"]}
out=[]
for d,base,bname in (("b2","base_qwen15.json","Qwen1.5-MoE-A2.7B-Chat 14.3B"),("dsv2l","base_dsv2l.json","DeepSeek-V2-Lite-Chat 15.7B"),("olmoe","base_olmoe.json","OLMoE-1B-7B-0125-Instruct 6.9B")):
    B=R(f"results/{d}/{base}"); bacc=sum(v[0] for v in B.values())/len(B)
    for f in sorted(glob.glob(f"results/{d}/*.json")):
        n=os.path.basename(f)
        if n.endswith(".size.json") or n.startswith(("base","stats","pre_")): continue
        try: sz=json.load(open(f.replace(".json",".size.json")))
        except Exception: continue
        C=R(f); keep=[k for k in B if B[k][0] and k in C]; a=sum(C[k][0] for k in keep)
        cells=collections.defaultdict(lambda:[0,0])
        for k in keep: c=cells[B[k][1]]; c[0]+=C[k][0]; c[1]+=1
        mc=min([(x/y,c) for c,(x,y) in cells.items() if y>=20] or [(float('nan'),None)])
        c=sz["cfg"]; xb=c["xb"]; kind=("GPTQ(gray) " if c.get("gptq") else "RTN ")+f"{xb}b g{c['xg']} experts, crit={c.get('crit')}, norm_topk={bool(c.get('norm'))}, attn {c['ab']}b/emb {c['eb']}b"
        out.append((bname,kind,sz["K"],sz["ne"],sz["ratio"]*100,bacc,a/len(keep),wl_(a,len(keep)),mc[0],mc[1],len(keep),n))
print("| method | base (base acc) | experts kept | size % of fp16 (analytic packed bytes) | retention [Wilson lo] n | min-cell ret (n>=20) | feedback | rule gaps | weakest rule | notes |")
print("|---|---|---|---|---|---|---|---|---|---|")
for o in sorted(out,key=lambda r:(r[0],-r[6])):
    print(f"| {o[1]} | {o[0]} ({o[5]:.3f}) | {o[2]}/{o[3]} | {o[4]:.2f}% | {o[6]:.3f} [{o[7]:.3f}] n={o[10]} | {o[8]:.3f} {o[9]} | n/a | R2,R3,R5,R6 (no locked exam, retention<98%) | R2 same-means-faithful (ret {o[6]*100:.0f}% < 98%) | dev SKILL7 (MASSIVE+CLINC150), real greedy GEN; tag {o[11]} |")
