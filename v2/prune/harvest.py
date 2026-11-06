import json,glob,subprocess,os,sys
os.makedirs("results",exist_ok=True)
subprocess.run(["modal","volume","get","--force","tenpct-prune","/","results/"],capture_output=True)
rs=[]
for f in sorted(glob.glob("results/*.json")):
    if os.path.basename(f).startswith("batch_") or "smoke" in f: continue
    try: rs.append(json.load(open(f)))
    except Exception: pass
rs=[r for r in rs if "acc" in r]
rs.sort(key=lambda r:(r["name"]))
print(f"{'name':26s} {'acc':>6s} {'agree':>6s} {'ratio_ent':>9s} {'ratio_nom':>9s} bpw")
for r in rs: print(f"{r['name']:26s} {r['acc']:6.4f} {r['agree']:6.4f} {r['ratio_ent']:9.4f} {r['ratio_nom']:9.4f} {r['bpw_body']:.2f} {r.get('dropped') or ''}")
for r in glob.glob("results/*delta*.json"): print(open(r).read()[:1500])
