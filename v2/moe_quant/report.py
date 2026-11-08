"""python report.py -> prints markdown tables (skill7, bank8, proxy) from results/"""
import json, glob, subprocess, sys, os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
for ds in ("skill7", "bank8"):
    if not glob.glob(f"results/outs_{ds}_base_SEL.json"): continue
    print(f"\n### {ds} (retention of base-correct items, real greedy generation; sorted by retention)\n"); print(subprocess.run([sys.executable, "score.py", "results", ds], capture_output=True, text=True).stdout)
P = {}
for f in sorted(glob.glob("results/proxy_*.json")): P.update(json.load(open(f)))
if P:
    print("\n### layer proxy (H-weighted relative error, 9 experts x 2 matrices from layers 6/24/42; lower better)\n\n| method | bpw (all bits) | rel err | secs | note |\n|---|---|---|---|---|")
    for k, v in sorted(P.items(), key=lambda kv: (round(kv[1]["bpw"] * 2) / 2 if kv[1]["bpw"] else 99, kv[1]["relerr"])): print(f"| {k} | {v['bpw']:.3f} | {v['relerr']:.4f} | {v['secs']} | {v['err'] or ''} |")
