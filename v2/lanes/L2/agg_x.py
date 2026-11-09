"""agg_x.py OUT.jsonl [teachkey]: mean over draws per (job, arm): teach, neigh, min_cell, new_on_neigh, MB."""
import json, sys, collections
R = [json.loads(l) for l in open(sys.argv[1])]; G = collections.OrderedDict()
for r in R: G.setdefault((r["job"], r["arm"]), []).append(r)
for (j, a), v in G.items():
    tk = [k for k in v[0] if k.startswith("teach_")]
    m = lambda k: sum(x[k] for x in v) / len(v)
    print(f"{j:18s} {a:12s} n={len(v)} " + " ".join(f"{k[6:]} {m(k):.3f}({min(x[k] for x in v):.2f}-{max(x[k] for x in v):.2f})" for k in tk) + f" | neigh {m('neigh'):.4f} cell {m('min_cell'):.3f} fp {m('new_on_neigh'):.4f} MB {(v[0].get('bytes') or 0)/1e6:.1f}")
