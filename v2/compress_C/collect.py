import json,subprocess,os,glob,shutil
d="resv"; shutil.rmtree(d,ignore_errors=True); os.makedirs(d,exist_ok=True)
subprocess.run(["modal","volume","get","--force","tenpct-C-res","/",d],capture_output=True)
rows={}
for f in glob.glob(d+"/**/*.json",recursive=True):
    try: j=json.load(open(f)); rows[j["res"]["name"]]=j
    except Exception: pass
base={}
for n,j in rows.items():
    if n.endswith("_base"): base[n[:-5]]=j
for n,j in sorted(rows.items()):
    r=j["res"]; k=n.split("_")[0]; b=base.get(k)
    ag=""
    if b and "ok" in b["res"]:
        ag=" agree_out=%.3f"%(sum(x==y for x,y in zip(j["outs"],b["outs"]))/len(j["outs"]))
    print(f'{n:24s} exact={r["exact"]:.3f} name={r["name_match"]:.3f} ratio={r["ratio_bf16"]*100:.2f}% bpw={r["body_bpw"]:.2f} MB={r["bytes"]/1e6:.0f} drop={len(r["dropped"])}{ag}')
