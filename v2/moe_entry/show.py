import sys,json,subprocess
out=subprocess.run(["modal","app","logs","tenpct-moe-entry"],capture_output=True,text=True).stdout+""
seen=set()
for l in out.splitlines():
    if not l.startswith("RESULT_JSON"): continue
    r=json.loads(l.split(" ",1)[1])
    if r["name"].endswith("_fb"): print("FB",r["name"],json.dumps({k:r[k] for k in ("empty",)} )[:600]); continue
    for sn,s in r["sets"].items():
        k=(r["name"],r["cfg"].get("ds","news"),sn)
        if k in seen: continue
        seen.add(k); print(r["cfg"].get("ds","news"),r["name"],sn,"ratio",round(r["bytes"]["ratio"],4),"acc",round(s["acc"],4),"base",round(s.get("base_acc",0),4),"ret",round(s.get("ret",0),4),"lo",round(s.get("ret_lo",0),4),"mincell",round(s.get("min_cell",0),3),s.get("flip_outs"))
