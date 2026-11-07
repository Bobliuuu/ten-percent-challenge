import json, glob, collections, numpy as np
out = {}
for f in sorted(glob.glob("results/probe/*.json")):
    d = json.load(open(f)); rows = d["rows"]; m = d["model"].split("/")[-1]; o = {}
    by = collections.defaultdict(list); bys = collections.defaultdict(list)
    for r in rows: by[f'{r["source"]}/{r["queue"]}'].append(r); bys[r["source"]].append(r)
    o["acc_source"] = {s: [round(float(np.mean([r["correct"] for r in v])), 3), len(v)] for s, v in sorted(bys.items())}
    o["macro_source"] = {s: round(float(np.mean([np.mean([r["correct"] for r in v2]) for k, v2 in by.items() if k.startswith(s + "/")])), 3) for s in bys}
    o["cells"] = {k: [round(float(np.mean([r["correct"] for r in v])), 3), len(v), round(float(np.mean([r["margin"] for r in v])), 2)] for k, v in sorted(by.items())}
    o["unparsed"] = sum(r["pred"] is None for r in rows)
    # confusion top errors
    ce = collections.Counter((r["source"], r["queue"], r["pred"]) for r in rows if not r["correct"]); o["top_errors"] = [[list(k), n] for k, n in ce.most_common(12)]
    # margin gating: acc among top-50% margin(top1 first-token lp)
    o["conf_gate"] = {s: [round(float(np.mean([r["correct"] for r in v if r["top1_lp"] > -0.05])), 3), sum(r["top1_lp"] > -0.05 for r in v)] for s, v in bys.items()}
    out[m] = o
json.dump(out, open("results/probe_summary.json", "w"), indent=1)
for m, o in out.items():
    print(m, "acc", o["acc_source"], "macro", o["macro_source"], "unparsed", o["unparsed"])
