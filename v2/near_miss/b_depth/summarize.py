import json,glob,subprocess,os,sys
P="results/Qwen2.5-14B-Instruct"
files=[f for f in sorted(glob.glob(P+"_*.gen.json")) if not f.endswith("_base.gen.json")]
out=subprocess.run([sys.executable,"analyze_nm.py",P+"_base.gen.json"]+files,capture_output=True,text=True,env=dict(os.environ,OUT="analysis.json")).stdout
rows=[]
for l in out.splitlines():
    if l.startswith("BASE"): print(l); continue
    try: r=json.loads(l)
    except: continue
    n=r["file"].split("Instruct_")[1].replace(".gen.json","")
    b=json.load(open(f"{P}_{n}.bytes.json"))["bytes"]
    g=lambda k:(round(r[k][0],3),r[k][1],r[k][2]) if k in r else None
    print(f"{n:28s} size {b['size_pct']:6.2f}% L{b['layers_kept']:2d} | EASY {g('ret_easy')} T1 {g('ret_t1')} TOX {g('ret_tox')} INJ {g('ret_inj')} ALL {g('ret_ALL')} | wc {r.get('worst_cells_None',[[None]])[0] if r.get('worst_cells_None') else ''}")
