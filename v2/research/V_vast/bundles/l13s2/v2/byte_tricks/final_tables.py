#!/usr/bin/env python3
"""final_tables: merge all results/L3*/*.jsonl, de-duplicate by (task,arm), print markdown with TOTAL-artifact pp: linear idx + best-coded scales (+emb/head if the arm quantized them) with other tensors fp16."""
import json, glob, os
HERE = os.path.dirname(os.path.abspath(__file__)); rows = {}
for f in sorted(glob.glob(os.path.join(HERE, "results", "L3*", "q14_*.jsonl"))):
    for l in open(f):
        r = json.loads(l)
        if not r.get("error"): rows[(r["task"], r["arm"])] = r
for task in ("SKILL7", "BANK6"):
    R = [r for (t, a), r in rows.items() if t == task]
    if not R: continue
    print(f"\n#### {task} (n base-correct {R[0]['score']['n_base_correct']}, base acc {R[0]['base_acc']:.3f})\n")
    print("| arm | idx bpw | scale bpw as coded | linear body pp of base | total artifact pp (emb/head fp16 unless arm trims them) | retention [Wilson lo] | min cell |\n|---|---|---|---|---|---|---|")
    for r in sorted(R, key=lambda r: r["arm"]):
        s = r["score"]; ret = f"{100*s['retention']:.1f} [{100*s['ret_lo']:.1f}]"
        if "lin_idx_bpw" in r:
            sb = r["lin_scale_bpw"]; c = r["cfg"]["coder"]; key = "fp16" if c in ("fp16", "fp32") else "rowmed"
            sc = sb.get(key, sb["fp16"]); lin = (r["lin_idx_bpw"] + sc) / 16 * 100 * r["lin_n"] / r["nparams"]
            tot = r["pct_by_coding"].get(key, r["pct_fp16_scales"]) if "pct_by_coding" in r else float("nan")
            print(f"| {r['arm']} | {r['lin_idx_bpw']:.3f} | {sc:.4f} ({key}) | {lin:.2f} | {tot:.2f} | {ret} | {s['min_cell']:.2f} |")
        else:
            tot = r.get("pct_total", float("nan")); eb, hb = r.get("emb_bits", 0), r.get("head_bits", 0)
            print(f"| {r['arm']} | - | - | (linear fp16) | emb+head {100*(eb+hb)/(16*r['nparams']):.2f} pp (fp16 baseline {100*2*778567680/r['nparams']:.2f}) | {ret} | {s['min_cell']:.2f} |")
