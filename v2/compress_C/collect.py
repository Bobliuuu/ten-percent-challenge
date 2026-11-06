import sys,json,subprocess,re
rows={}
for app in sys.argv[1:]:
    out=subprocess.run(["modal","app","logs",app],capture_output=True,text=True).stdout
    for l in out.splitlines():
        if l.startswith("RESULT_JSON "):
            r=json.loads(l[12:]); rows[r["name"]]=r
for n,r in sorted(rows.items()):
    print(f'{n:24s} exact={r["exact"]:.3f} name={r["name_match"]:.3f} ratio={r["ratio_bf16"]*100:.2f}% bpw={r["body_bpw"]:.2f} MB={r["bytes"]/1e6:.0f} drop={len(r["dropped"])} oov={r["oov"]:.3f}')
json.dump(rows,open("results.json","w"))
