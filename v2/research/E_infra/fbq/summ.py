import json, sys
for f in sys.argv[1:]:
    R = json.load(open(f)); print("==", f, R["model"], "k", R["k"], "sites", R["sites"], "omega", R["omega"], "concept", R["concept"], "ndraw", R["ndraw"], "nann", R["nann"])
    print("base acc on sets", {k: round(v, 3) for k, v in R["base_on_sets"].items()}, "spec_n", R["spec_n"], "spec", R["spec_sha256"][:12])
    for v, r in R["results"].items():
        g = r["gaps"]; a = r["arms"]
        f2 = lambda x: "%+.2f(%+.2f)" % (x["mean"], x["ci"][0])
        print(f"{v:5s} teach1 gain={f2(g['real_minus_empty|teach1'])} -mixed={f2(g['real_minus_mixed|teach1'])} -randwrong={f2(g['real_minus_randwrong|teach1'])} | "
              f"teach3 gain={f2(g['real_minus_empty|teach3']) if 'real_minus_empty|teach3' in g else 'NA'} -mixed={f2(g['real_minus_mixed|teach3']) if 'real_minus_mixed|teach3' in g else 'NA'} | "
              f"nb ret real={a['real']['nb']['mean']:.3f} mixed={a['mixed']['nb']['mean']:.3f} rw={a['randwrong']['nb']['mean']:.3f} nb3={a['real']['nb3']['mean']:.3f} devret={a['real']['devret']['mean']:.3f} | bytes_changed real={r['bytes_changed']['real']:.0f}")
