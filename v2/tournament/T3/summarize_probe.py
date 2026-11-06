"""Build summary table from results/<model>.json. AUROC from first-token logit diagnostic where available."""
import json, glob, os, math
from sklearn.metrics import roc_auc_score
H = os.path.dirname(os.path.abspath(__file__)); rows = []; out = []
for f in sorted(glob.glob(H + "/results/*.json")):
    if f.endswith("summary.json"): continue
    r = json.load(open(f)); gb = r["nparams"] * 2
    for pn, p in r["prompts"].items():
        s = p["summary"]; it = p["items"]; d = {}
        for src in ("S1", "S3"):
            sub = [x for x in it if x["source"] == src]; d[src] = sum(x["correct"] for x in sub) / len(sub)
            sc = [(x["gold"] == "ai", x["ai_minus_human_logit"]) for x in sub if x["ai_minus_human_logit"] is not None]
            d["auc_" + src] = roc_auc_score([a for a, _ in sc], [b for _, b in sc]) if len(sc) == len(sub) and len(set(a for a, _ in sc)) == 2 else None
        ok = lambda a: 0.65 <= a <= 0.92
        rows.append(dict(model=r["model"], prompt=pn, acc=s["acc"], S1=d["S1"], S3=d["S3"], auc_S1=d["auc_S1"], auc_S3=d["auc_S3"], per_kind=s["per_kind"], fmt_fail=s["format_fail"], pred_dist=s["pred_dist"],
                         R8_S1=ok(d["S1"]), R8_S3=ok(d["S3"]), fp16_bytes=gb, budget10pct_bytes=gb * 0.10, mean_out_tokens=sum(x["out_tokens"] for x in it) / len(it)))
json.dump(rows, open(H + "/results/summary.json", "w"), indent=1)
f = lambda x: "  -  " if x is None else f"{x:.3f}"
print("| model | prompt | acc S1-dev | acc S3-dev | AUROC S1 / S3 | S1_h / S1_ai / S3_h / S3_ai | fmt-fail | pred dist | R8 S1 / S3 (65-92%) | fp16 GB | 10% GB |\n|---|---|---|---|---|---|---|---|---|---|---|")
for r in rows:
    k = r["per_kind"]
    print(f"| {r['model']} | {r['prompt']} | {r['S1']:.3f} | {r['S3']:.3f} | {f(r['auc_S1'])} / {f(r['auc_S3'])} | {k['S1_human']:.2f}/{k['S1_ai']:.2f}/{k['S3_human']:.2f}/{k['S3_ai']:.2f} | {r['fmt_fail']:.3f} | {r['pred_dist']} | {'Y' if r['R8_S1'] else 'n'} / {'Y' if r['R8_S3'] else 'n'} | {r['fp16_bytes']/1e9:.1f} | {r['budget10pct_bytes']/1e9:.2f} |")
