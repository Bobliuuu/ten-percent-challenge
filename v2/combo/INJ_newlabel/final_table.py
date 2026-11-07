import json,glob,analyze,sys,collections
src={"Qwen3-4B":["results5/Qwen3-4B.json"],"Qwen3-8B":["results/Qwen3-8B.json"],"Qwen3-14B":["results/Qwen3-14B.json"],"Qwen3-32B":["results3/Qwen3-32B.json","results4/Qwen3-32B.json"]}
rows=[]
for mdl,fs in src.items():
    for k,f in enumerate(fs):
        try: R=json.load(open(f))
        except Exception: continue
        for m in sorted({r["m"] for r in R["runs"]}):
            if k>0 and m!="lda": continue
            tag=m if k==0 else "lda_fine"
            for n in sorted({r["n"] for r in R["runs"] if r["m"]==m}):
                try: c=analyze.cell(R,m,n)
                except Exception as e: continue
                c["m"]=tag; c["alpha"]=R.get("alphas",{}).get(m); rows.append(c)
json.dump(rows,open("final_cells.json","w"),indent=1)
f=lambda x:"%+.2f(%+.2f)"%(x[0],x[1])
out=["| model | method | alpha | n | draws | teach real | gain (lo) | r-wrong | r-mixed_wrong | r-mixed_own | r-shuffle | neigh | Wilson lo | min cell (src/class) | OOD gain (lo) | G2 all | G2 excl mixed_own |","|"+"---|"*17]
for o in rows:
    if "_f" in o["m"] and o["m"].startswith("plain_") : pass
    g=lambda k:f(o[k]) if k in o else "-"
    mc=min(o["neigh_by"].items(),key=lambda kv:kv[1])
    out.append(f"| {o['model']} | {o['m']} | {o.get('alpha') or ''} | {o['n']} | {o['draws']} | {o['teach_real']:.2f} | {g('gain')} | {g('r-wrong')} | {g('r-mixed_wrong')} | {g('r-mixed_own')} | {g('r-shuffle')} | {o['neigh']:.3f} | {o['neigh_wlo']:.3f} | {mc[0]} {mc[1]:.2f} | {g('ood_gain')} | {'PASS' if o['G2'] else 'fail'} | {'PASS' if o['G2_noOwn'] else 'fail'} |")
open("final_table.md","w").write("\n".join(out)+"\n"); print("\n".join(out))
# summary: n where all controls pass per model/method
print()
for mdl in src:
    for m in sorted({o["m"] for o in rows if o["model"]==mdl}):
        ns=[o["n"] for o in rows if o["model"]==mdl and o["m"]==m]
        p=[o["n"] for o in rows if o["model"]==mdl and o["m"]==m and o["G2"]]; q=[o["n"] for o in rows if o["model"]==mdl and o["m"]==m and o["G2_noOwn"]]
        if len(ns)>=3: print(mdl,m,"G2 pass n:",p,"| excl mixed_own:",q)
