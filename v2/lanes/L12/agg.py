"""aggregate results/vast*/res/*.json -> markdown table (SEL = selection set, REP = report set)"""
import json, glob, sys, os
rows = []
for f in sorted(glob.glob("results/*/res/*.json")):
    n = os.path.basename(f)
    if n.startswith(("base_", "outs_", "stats_", "loo_", "lsens_", "fb_")): continue
    r = json.load(open(f)); s = r["sets"].get("sel", {}); p = r["sets"].get("rep", {})
    rows.append((r["name"], r["bytes"]["ratio"], s, p, r["cfg"]))
rows.sort(key=lambda x: x[0])
print("| config | size/fp16 | SEL acc | SEL ret [lo] | SEL min-cell | REP ret [lo] | REP min-cell (lo) |\n|---|---|---|---|---|---|---|")
for n, ratio, s, p, cfg in rows:
    g = lambda d, k: d.get(k, float("nan"))
    print(f"| {n} | {ratio*100:.2f}% | {g(s,'acc'):.3f} | {g(s,'ret'):.3f} [{g(s,'ret_lo'):.3f}] | {g(s,'min_cell'):.3f} | {g(p,'ret'):.3f} [{g(p,'ret_lo'):.3f}] | {g(p,'min_cell'):.3f} ({g(p,'min_cell_lo'):.3f}) |")
