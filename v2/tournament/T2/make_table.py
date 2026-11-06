import json,glob,os
H=os.path.dirname(os.path.abspath(__file__))
FP16={"Qwen3-30B-A3B-Instruct-2507":30.5e9,"Qwen2.5-32B-Instruct":32.76e9,"Qwen3-14B":14.77e9,"Qwen3-8B":8.19e9,"gpt-oss-20b":20.9e9}
L=["| model | fp16-eq GB | 10% budget GB | S1 acc | S3 acc | all | nopii | one | few | many | fmt-fail | untyped F1 | R8 (60-92%) |","|"+"---|"*13]
for f in sorted(glob.glob(H+"/results/*.json")):
    r=json.load(open(f)); s=r["summary"]; m=r["model"]; b=FP16.get(m,0)*2/1e9
    g=lambda k:s[k]["acc"]*100 if k in s else float("nan")
    ok="yes" if all(60<=g(f"source={x}")<=92 for x in("S1","S3")) else "NO"
    L.append(f"| {m} | {b:.1f} | {b/10:.2f} | {g('source=S1'):.1f} | {g('source=S3'):.1f} | {g('all'):.1f} | {g('kind=nopii'):.1f} | {g('kind=one'):.1f} | {g('kind=few'):.1f} | {g('kind=many'):.1f} | {s['all']['fmt_fail']*100:.1f}% | {s['all']['untyped_f1']*100:.1f} | {ok} |")
open(H+"/results/SUMMARY.md","w").write("\n".join(L)+"\n"); print("\n".join(L))
