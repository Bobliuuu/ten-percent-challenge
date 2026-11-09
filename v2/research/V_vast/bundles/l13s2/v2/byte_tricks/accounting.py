#!/usr/bin/env python3
"""accounting: analytic byte savings (percent of base fp16) per trick per start model, from MEASURED tensor statistics in results/stats/*.json (+ stats_embed).
Prints markdown. saving% = bits_saved / (16 * params_total) * 100."""
import json, glob, os, statistics as st
HERE = os.path.dirname(os.path.abspath(__file__)); S = {}
for f in glob.glob(os.path.join(HERE, "results", "stats", "stats_*.json")):
    r = json.load(open(f))
    if "model" in r and "family_params" in r: S[r["model"]] = r
def mean(xs): return sum(xs) / len(xs)
def lin_stat(r, key):
    xs = [x[key] for x in r["linear"] if key in x]
    return None if not xs else (mean([x["idx_bpw"] for x in xs]), {k: mean([x["scale_bpw"][k] for x in xs]) for k in xs[0]["scale_bpw"]})
print("| model | linear/expert % of params | emb+head % | scales fp16 g128 -> coded (bpw) | save % base | repo 11-bit -> coded save % | ANS vs bit-pack (L5, 3b->H0) save % | vocab trim to 65536 (emb/head at 4 b) save % | tokenizer dedup+xz save % | modality removal % |")
print("|---|---|---|---|---|---|---|---|---|---|")
for m, r in S.items():
    P = r["params_total"]; fp = r["family_params"]; lin = (fp.get("linear", 0) + fp.get("expert", 0)); lf = lin / P
    eh = (fp.get("embed", 0) + fp.get("head", 0)) / P
    ls = lin_stat(r, "L5_g128_s1")
    row = [m, f"{100*lf:.1f}", f"{100*eh:.2f}"]
    if ls:
        idx, sc = ls; coded = min(sc["H0"], sc["row+col"], sc["rowmed"]); save = (sc["fp16"] - coded) / 16 * lf * 100; save11 = (11 / 128 - coded) / 16 * lf * 100
        ans = (3 - idx) / 16 * lf * 100
        row += [f"{sc['fp16']:.3f} -> {coded:.4f}", f"{save:.3f}", f"{save11:.3f}", f"{ans:.2f}"]
    else: row += ["n/a (discrete model, see below)"] * 4
    vs = 100 * eh * (1 - 65536 / 151936) * (4 / 16)
    sf = r["small_files"]; tok = sf.get("tokenizer.json", {}).get("bytes", 0); allt = sum(v["bytes"] for k, v in sf.items() if k in ("tokenizer.json", "vocab.json", "merges.txt")); xz = sf.get("tokenizer.json", {}).get("xz", 0)
    tsave = 100 * (allt - xz) / (2 * P)
    row += [f"{vs:.2f}", f"{tsave:.4f}", f"{100*fp.get('modality', 0)/P:.2f}"]
    print("| " + " | ".join(row) + " |")
