import json,glob,sys,numpy as np,os
model=sys.argv[1]; R={os.path.basename(f)[len(f"ct_{model}_"):-5]:json.load(open(f)) for f in glob.glob(f"results/ct_{model}_*.json")}
base={r['id']:r for r in R['bf16']['rows']}; bins=[0,2,4,8,12,16,100]
print("retention by base first-token margin bin (all tasks pooled except AIDET), n in parens")
print("cfg | "+" | ".join(f"[{a},{b})" for a,b in zip(bins,bins[1:])))
for c in ["t096","t15","t20","t30"]:
    if c not in R: continue
    cr={r['id']:r for r in R[c]['rows']}; cells=[]
    for a,b in zip(bins,bins[1:]):
        ids=[i for i,r in base.items() if r['correct'] and r['task']!='AIDET' and a<=r['m_min']<b]
        cells.append(f"{np.mean([cr[i]['correct'] for i in ids]) if ids else float('nan'):.2f} ({len(ids)})")
    print(c," | "," | ".join(cells))
# margin shift stats: per-item noise in first-token margin (cand - base), base-correct items
for c in ["t096","t15","t20","t30"]:
    if c not in R: continue
    cr={r['id']:r for r in R[c]['rows']}
    d=np.array([cr[i]['m_first']-base[i]['m_first'] for i in base if base[i]['correct'] and base[i]['task']!='AIDET'])
    print(c,"delta margin mean %.2f std %.2f  (median base margin ~%.1f)"%(d.mean(),d.std(),np.median([r['m_first'] for r in base.values() if r['correct']])))
