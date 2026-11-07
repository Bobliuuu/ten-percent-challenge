"""Summarise results/*.json base probes (dev only): per (source x kind[/hard]) accuracy with Wilson CI, balanced accuracy per split, S1->S3 gap on shared classes,
confusion matrices, first-token margin stats, rule-8 flags (Wilson lo > 0.30 [chance 0.20, 5-way] and hi < 0.98 [ceiling]). Writes results/summary.md + summary.json"""
import json, glob, os, collections, numpy as np, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import scorer
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
out = {}; md = ["# TOX base probe (dev only; greedy; fixed prompt scorer.PROMPT; strict parse)\n"]
for f in sorted(glob.glob(f"{R}/*.json")):
    b = os.path.basename(f)
    if b.startswith(("summary", "baselines")) or b.endswith("_spec.json") or "fb" in b: continue
    d = json.load(open(f)); rows = d["items"]; m = d["model"]; out[m] = {}
    md.append(f"\n## {m}\n"); md.append("| split | kind | n | acc | 95% CI | parse-fail | median first-token margin(gold-best other, logprob) | rule-8 |\n|---|---|---|---|---|---|---|---|")
    cells = collections.defaultdict(list)
    for r in rows:
        k = (r["split"], r["kind"] + ("/hard" if r["kind"] == "safe" and r["hard"] else "/plain" if r["kind"] == "safe" else ""))
        cells[k].append(r); cells[(r["split"], "ALL-cells(macro)")].append(r)
    bal = {}
    for (sp, k), rs in sorted(cells.items()):
        n = len(rs); a = sum(r["pred"] == r["gold"] for r in rs); lo, hi = scorer.wilson(a, n); pf = np.mean([r["pred"] is None for r in rs])
        mg = np.median([r["lp"][r["gold"]] - max(v for l, v in r["lp"].items() if l != r["gold"]) for r in rs])
        flag = "" if k.startswith("ALL") else ("OK" if (lo > .30 and hi < .98) else ("CEILING" if hi >= .98 else "LOW"))
        md.append(f"| {sp} | {k} | {n} | {a/n:.3f} | [{lo:.3f},{hi:.3f}] | {pf:.3f} | {mg:.1f} | {flag} |"); out[m][f"{sp}|{k}"] = dict(n=n, acc=a / n, lo=lo, hi=hi, parse_fail=float(pf), flag=flag)
    for sp in ("s1_dev", "s3_dev", "s2_probe"):
        by = collections.defaultdict(list)
        for r in rows:
            if r["split"] == sp: by[r["gold"]].append(r["pred"] == r["gold"])
        bal[sp] = float(np.mean([np.mean(v) for v in by.values()])); out[m][f"{sp}|balanced_acc"] = bal[sp]
    md.append(f"\nbalanced accuracy (macro over gold classes): " + ", ".join(f"{k}={v:.3f}" for k, v in bal.items()))
    # S1->S3 gap on shared classes (safe, toxic, hate)
    def shared(sp): 
        by = collections.defaultdict(list)
        for r in rows:
            if r["split"] == sp and r["gold"] in ("safe", "toxic", "hate"): by[r["gold"]].append(r["pred"] == r["gold"])
        return float(np.mean([np.mean(v) for v in by.values()]))
    md.append(f"\nS1->S3 gap on shared classes (macro safe/toxic/hate): S1={shared('s1_dev'):.3f}  S3={shared('s3_dev'):.3f}  gap={shared('s1_dev')-shared('s3_dev'):+.3f};  S1->S2probe (macro over S2 classes present): S1={shared('s1_dev'):.3f}  S2={shared('s2_probe'):.3f}")
    for sp in ("s1_dev", "s3_dev", "s2_probe"):
        C = collections.defaultdict(collections.Counter)
        for r in rows:
            if r["split"] == sp: C[r["gold"]][r["pred"] or "NONE"] += 1
        md.append(f"\nconfusion {sp} (gold -> predicted counts):\n```\n" + "\n".join(f"{g:9s} " + " ".join(f"{p}:{c}" for p, c in sorted(cs.items(), key=lambda x: -x[1])) for g, cs in sorted(C.items())) + "\n```")
open(f"{R}/summary.md", "w").write("\n".join(md)); json.dump(out, open(f"{R}/summary.json", "w"), indent=1); print("\n".join(md))
