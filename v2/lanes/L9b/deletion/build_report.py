import json, os
cfgs = [("c", "klw 1 (dev: K .0005%)", 0.9), ("p", "klw 10", 0.8), ("q", "klw 30", 0.75)]
rows = []; ex = []
for tag, desc, gate in cfgs:
    f = f"results/locked_{tag}_summary.json"
    if not os.path.exists(f): continue
    s = json.load(open(f)); C = s["conds"]; e = s["empty"]; r, sh, w = C["real"], C["shuffled"], C.get("wrong", {})
    rows.append(f"| {tag}: {desc} | {r['n']} | {r['teach_mean']:.3f} ({r['teach_min']:.2f}-{r['teach_max']:.2f}) | {r['neigh_mean']:.3f} ({r['neigh_min']:.3f}-) | {r['min_cell_mean']:.3f} | {sh['teach_mean']:.3f} ({sh['teach_min']:.2f}-{sh['teach_max']:.2f}) | {w.get('teach_mean', float('nan')):.3f} | {r['n_deleted_mean']} ({100*r['frac_deleted_mean']:.4f}%) | {r['carrier_bytes_mean']} |")
    rows_g = f"| {tag} | {gate} | {r['gated_accepted']}/{r['n']} | {sh['gated_accepted']}/{sh['n']} | {w.get('gated_accepted', 'NA')}/{w.get('n', 'NA')} | {r['fit_acc']} | {sh['fit_acc']} |"
    ex.append((tag, s["exam"], rows_g, e, s))
print("| config | draws | real teach mean (min-max) | neighbours kept real (min) | min cell | shuffled teach mean (min-max) | wrong teach | deleted units | carrier B |\n|---|---|---|---|---|---|---|---|---|")
print("\n".join(rows))
print("\nempty neighbours:", ex[0][3]["neigh"], "n", ex[0][3]["n_neigh"], "min_cell", ex[0][3]["min_cell"])
print("\ngate table\n| cfg | gate | real accepted | shuffled accepted | wrong accepted | real fit | shuffled fit |\n|---|---|---|---|---|---|---|")
for t, e, g, _, _ in ex: print(g)
print("\nexam retention after feedback (real arm, draws 0-1, 2744 items, 2655 base-correct)")
for t, e, g, _, s in ex: print(t, [(round(x["ret"], 4), round(x["ret_lo"], 4), x["worst_kind"], x["worst_cell"]) for x in e])
