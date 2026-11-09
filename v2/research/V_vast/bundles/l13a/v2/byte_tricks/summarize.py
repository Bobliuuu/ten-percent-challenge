#!/usr/bin/env python3
"""summarize results/*.jsonl -> markdown tables (arm, bits/weight idx+scales, % of base fp16 for each scale coding, retention + Wilson low, min cell)."""
import json, sys, glob, os
HERE = os.path.dirname(os.path.abspath(__file__))
def rows(path):
    return [json.loads(l) for l in open(path)]
def main():
    files = sys.argv[1:] or sorted(glob.glob(os.path.join(HERE, "results", "**", "q14_*.jsonl"), recursive=True))
    for f in files:
        R = rows(f); print(f"\n### {os.path.relpath(f, HERE)}  (base acc {R[0].get('base_acc')})\n")
        print("| arm | idx bpw | scale bpw (fp16 / best coded) | linear+scale % of base fp16 | retention | Wilson low | min cell | note |\n|---|---|---|---|---|---|---|---|")
        for r in R:
            if r.get("error"): print(f"| {r['arm']} | ERR {r['error'][:60]} |||||||"); continue
            s = r["score"]; sb = r.get("lin_scale_bpw", {}); best = min(sb.values()) if sb else None
            lin = (r["lin_idx_bpw"] + (best if best is not None else 0)) / 16 * 100 * r["lin_n"] / r["nparams"] if "lin_idx_bpw" in r else None
            note = ""
            if r.get("vocab"): note = f"retok {r['vocab']['replaced']}/{r['vocab']['total']} labels_kept={r['vocab']['label_tokens_kept']}"
            if r.get("levels_hist"): note = str(r["levels_hist"])
            tot = r.get("pct_by_coding") or r.get("pct_total")
            print(f"| {r['arm']} | {r.get('lin_idx_bpw', float('nan')):.3f} | {sb.get('fp16', float('nan')):.4f} / {best if best is not None else float('nan'):.4f} | {lin if lin is not None else float('nan'):.3f}% | {s['retention']:.4f} | {s['ret_lo']:.4f} | {s['min_cell']:.3f} | {note} |")
if __name__ == "__main__": main()
