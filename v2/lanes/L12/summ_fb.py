"""Summarise FBO/FBS lines from a log file: python3 summ_fb.py log [FBO|FBS]"""
import sys, json, collections
tag = sys.argv[2] if len(sys.argv) > 2 else "FBS"; g = collections.defaultdict(list)
for l in open(sys.argv[1]):
    if f"{tag} {{" in l:
        r = json.loads(l[l.index("{"):]); k = (json.dumps(r.get("grid"), sort_keys=True) if tag == "FBS" else f"lam{r['lam']}", r["cond"]); g[k].append((r["teach_acc"], r["neigh_ret"], r["neigh_leak"], r.get("carrier_bytes")))
for k, v in sorted(g.items()):
    print(k[0], k[1], "n=%d teach %s neigh %s leak %s cb %s" % (len(v), [round(x[0], 3) for x in v], [round(x[1], 3) for x in v], [x[2] for x in v], v[0][3]))
