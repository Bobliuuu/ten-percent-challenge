import json,glob,collections,os,re,sys
sys.path.insert(0,".")
from scorer import score_item
ITEMS={json.loads(l)["id"]:json.loads(l) for l in open("probe_items.jsonl")}
def glm2qwen(t):
    def one(m):
        body=m.group(1); name=body.split("<arg_key>")[0].strip(); args={}
        for k,v in re.findall(r"<arg_key>(.*?)</arg_key>\s*<arg_value>(.*?)</arg_value>",body,re.S):
            try: args[k.strip()]=json.loads(v)
            except Exception: args[k.strip()]=v
        return "<tool_call>"+json.dumps({"name":name,"arguments":args})+"</tool_call>"
    return re.sub(r"<tool_call>(.*?)</tool_call>",one,t,flags=re.S)
ADAPT={"GLM-4.5-Air-FP8":glm2qwen,"GLM-4.5-Air":glm2qwen,"GLM-4.5-FP8":glm2qwen}
SZ={"Qwen3-235B-FP8":235.1,"gpt-oss-120b":116.8,"GLM-4.5-Air-FP8":110.5,"GLM-4.5-Air":110.5,"Qwen3-Coder-480B-FP8":480.2,"GLM-4.5-FP8":358.3,"Qwen3-Next-80B":81.3,"Mixtral-8x22B":140.6}
GPUS={"H100:4":4,"H100:2":2,"H100:2":2,"H100:8":8}
PR={"H100":3.95,"H200":4.54}
L=["| model | prec/GPU | bf16 GB | 10% GB | S1 all | S1 n/s/m | S3 all | S3 n/s/m | parse-fail | R8 | wall min | ~$ (excl. download) |","|---|---|---|---|---|---|---|---|---|---|---|---|"]
for f in sorted(glob.glob("results/*.json")):
    d=json.load(open(f)); R=d["rows"]; n=d["model"]
    if n in ADAPT:
        for r in R:
            sc=score_item(ADAPT[n](r["output"]),ITEMS[r["id"]]); r["correct"]=bool(sc["correct"]); r["parse_fail"]=bool(sc["parse_fail"])
        n=n+" (GLM-XML->json adapter)"
    def acc(rs): return sum(r["correct"] for r in rs)/max(1,len(rs))
    def pk(src):
        rs=[r for r in R if r["source"]==src]; k={}
        for kind in("nocall","single","multi"): k[kind]=acc([r for r in rs if r["kind"]==kind])
        return acc(rs),"%.2f/%.2f/%.2f"%(k["nocall"],k["single"],k["multi"])
    srcs=sorted({r["source"] for r in R}); a=[pk(s) for s in srcs]
    pf=sum(r["parse_fail"] for r in R)/len(R)
    r8=all(.6<=x[0]<=.92 for x in a)
    g,c=d["gpu"].split(":"); cost=int(c)*PR[g]*d["secs"]/3600
    L.append(f"| {n} | {d['precision']} {d['gpu']} | {SZ.get(d["model"],'?')} | {SZ.get(d["model"],0)/10:.1f} | {a[0][0]:.3f} | {a[0][1]} | {a[1][0]:.3f} | {a[1][1]} | {pf:.3f} | {'YES' if r8 else 'no'} | {d['secs']/60:.0f} | {cost:.1f} |")
    print(n,srcs)
open("BASE.md","w").write("# L2 MoE base tool-calling probe (greedy, native tools template, same scorer/600 items)\n\n"+"\n".join(L)+"\n")
print("\n".join(L))
