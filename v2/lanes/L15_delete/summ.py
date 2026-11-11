"""print a compact table of an L15 results JSON:  python summ.py results/X.json ..."""
import json, sys
for p in sys.argv[1:]:
    r = json.load(open(p)); print("==", p, r.get("cfgs"), "n_dev_neigh", r.get("n_dev_neigh"), "n_neigh", r.get("n_neigh_base_correct"), "base_exam_acc", r.get("base_exam_acc"))
    if r.get("empty"): print("   empty", {k: r["empty"][k] for k in ("teach", "neigh_kept")})
    for x in r["runs"]:
        t = x.get("dev_teach", x.get("teach")); n = x.get("dev_neigh_kept", x.get("neigh_kept")); ex = x.get("exam_after", {}).get("overall", {}).get("ret")
        print(f"   d{x['d']} {x['arm']:8s} teach {t:.3f} neigh {n if n is None else round(n,3)} exam_ret {ex if ex is None else round(ex,4)} del {x['n_deleted']} ({100*x['frac_of_target']:.3f}% of gate) new {x['deleted_not_in_prior']} bytes {x['bytes']} s {x['seconds']} out {x.get('dev_teach_out', x.get('teach_out'))[:3]}")
