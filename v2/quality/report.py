import json, glob, os, numpy as np, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze import ret
bases = {"7b": json.load(open("results/base7b.json")), "32b": json.load(open("results/base32b.json")) if os.path.exists("results/base32b.json") else None}
rows = []
for f in sorted(glob.glob("results/*.json")):
    n = os.path.basename(f)[:-5]
    if n.startswith(("base", "sens")): continue
    c = json.load(open(f)); b = bases["32b" if n.startswith("q32") else "7b"]
    if "real_inj" not in c or b is None: continue
    cell = lambda r: [str(r["cell"][0]) + "_" + str(r["cell"][1])]
    a = ret(b["real_inj"], c["real_inj"], cell); t = ret(b["real_toolsel"], c["real_toolsel"], cell)
    gs = c.get("general_summary") or {k: dict(retention=float(((np.array(v) == np.array(c["general"]["y"][k])) & (np.array(b["general"]["preds"][k]) == np.array(b["general"]["y"][k]))).sum() / (np.array(b["general"]["preds"][k]) == np.array(b["general"]["y"][k])).sum())) for k, v in c["general"]["preds"].items()}
    rows.append(f"| {n} | {c['ratio_fixed']*100:.2f} | {c['body_bpw_fixed']:.3f} | {a[0]*100:.1f} ({a[1]*100:.1f}) / {a[2]*100:.0f} | {t[0]*100:.1f} ({t[1]*100:.1f}) / {t[2]*100:.0f} | {'/'.join(str(round(v['retention']*100)) for v in gs.values())} | {c['general']['ppl']:.2f} | {c['general']['kl']:.3f} |")
print("| run | size % (fixed-length idx) | body bpw | INJ-real retention (Wilson lo) / worst cell | TOOLSEL-real retention (lo) / worst cell | MMLU/ARC/HS/TQA retention | wikitext ppl | KL |\n|---|---|---|---|---|---|---|---|")
print("\n".join(rows))
