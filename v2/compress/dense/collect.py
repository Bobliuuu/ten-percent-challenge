import json,glob,os,numpy as np
rows=[]
for f in sorted(glob.glob("results/res_*.json")):
    r=json.load(open(f)); c=r["cfg"]; n=os.path.basename(f)[4:-5]
    I=r["id"]["gen"]["ALL"]; O=r["ood"]["gen"]["ALL"]
    rows.append(dict(run=n,model=r["model"].split("/")[-1],bytes=int(r["bytes"]),ratio=round(r["ratio"],4),bpw=round(r["body_bpw"],2),dropped=len(r["dropped"]),
      a_id=I["exact64"],a16_id=I["exact16"],pfx_id=I["prefix_frac"],b_id=r["id"]["tf_top1"],c_id=r["id"]["tf_kl"],
      a_ood=O["exact64"],b_ood=r["ood"]["tf_top1"],c_ood=r["ood"]["tf_kl"],floor=r["base_noise_floor"],
      ood_drop_a=I["exact64"]-O["exact64"],ood_drop_b=r["id"]["tf_top1"]-r["ood"]["tf_top1"],quick=bool(c.get("qstride"))))
json.dump(rows,open("results_table.json","w"),indent=1)
hdr="| run | bytes | ratio | body bpw | drop | (a) exact64 ID | exact16 | (b) top1 | (c) KL | OOD (a) | OOD (b) | OOD (c) | OOD drop a/b | bf16 self-floor |\n|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n"
t=hdr
for r in rows:
    if r["quick"]: continue
    t+=f"| {r['run']} | {r['bytes']:,} | {r['ratio']} | {r['bpw']} | {r['dropped']} | {r['a_id']:.3f} | {r['a16_id']:.3f} | {r['b_id']:.3f} | {r['c_id']:.3f} | {r['a_ood']:.3f} | {r['b_ood']:.3f} | {r['c_ood']:.3f} | {r['ood_drop_a']:+.3f}/{r['ood_drop_b']:+.3f} | {r['floor']} |\n"
open("results_table.md","w").write(t); print(t)
q=[r for r in rows if r["quick"]]
print("quick-stage runs:",[(r["run"],round(r["c_id"],3),round(r["b_id"],3)) for r in q])
