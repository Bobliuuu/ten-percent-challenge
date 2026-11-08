"""Margin-tournament analysis. Input: base_<model>.json (per-item rows from tour_base.py). Output: tour_tables.json + RESULTS_BASE.md.
Predicted retention = mean over base-correct items of 1 - flip(bin(m_min)), flip curves from e0_tables.json (our measured 32B / 14B flips at 1.57/1.79/1.96 bpw, pooled over 12 tasks).
Wilson-sized pass probability: P(every cell lower bound >= 0.95 and overall >= 0.98) via binomial simulation at the predicted per-cell retention."""
import json, sys, os, math, collections, numpy as np
H = os.path.dirname(os.path.abspath(__file__)); D = sys.argv[1] if len(sys.argv) > 1 else H + "/results"
E0 = json.load(open(H + "/e0_tables.json"))["margin"]
EDGE = [0, 2, 4, 6, 8, 10, 12, 16, 99]
def curve(model, cfg):
    k = "32B pooled 12 tasks" if "32B" in model else "14B pooled 12 tasks"
    return [r["flip_" + cfg] for r in E0[k]]
def flip_p(cv, m):
    b = max(j for j in range(len(EDGE) - 1) if EDGE[j] <= max(m, 0)); return cv[min(b, len(cv) - 1)]
def wil(k, n, z=1.96):
    if n == 0: return (0, 1)
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); a = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)); return ((c - a) / d, (c + a) / d)
OUT = {}; rng = np.random.default_rng(0)
for f in sorted(os.listdir(D)):
    if not f.startswith("base_"): continue
    d = json.load(open(f"{D}/{f}")); mod = d["model"].split("/")[-1]; rows = d["rows"]
    by = collections.defaultdict(list)
    for r in rows: by[r["task"]].append(r)
    for t, rs in sorted(by.items()):
        bc = [r for r in rs if r["correct"]]; n = len(bc)
        if n == 0: continue
        e = dict(n=len(rs), base_acc=n / len(rs), n_bc=n)
        for thr in (4, 8, 12): e[f"frac_m{thr}"] = float(np.mean([r["m_min"] >= thr for r in bc]))
        e["median_m"] = float(np.median([r["m_min"] for r in bc]))
        for cfg in ("r100", "r1125", "r125"):
            cv = curve(mod, cfg); e["pred_" + cfg] = float(np.mean([1 - flip_p(cv, r["m_min"]) for r in bc]))
        cv = curve(mod, "r100")
        cells = collections.defaultdict(list)
        for r in bc: cells[(r["src"], r["kind"])].append(1 - flip_p(cv, r["m_min"]))
        kc = collections.defaultdict(list)
        for r in bc: kc[r["kind"]].append(1 - flip_p(cv, r["m_min"]))
        e["kinds"] = {k: dict(n_bc=len(v), pred=float(np.mean(v)), base_acc=float(np.mean([x["correct"] for x in rs if x["kind"] == k]))) for k, v in kc.items()}
        e["srcs"] = {s: dict(base_acc=float(np.mean([x["correct"] for x in rs if x["src"] == s])), pred=float(np.mean([1 - flip_p(cv, x["m_min"]) for x in bc if x["src"] == s] or [0])), frac_m8=float(np.mean([x["m_min"] >= 8 for x in bc if x["src"] == s] or [0]))) for s in sorted(set(r["src"] for r in rs))}
        wk = min(e["kinds"].items(), key=lambda kv: kv[1]["pred"] if kv[1]["n_bc"] >= 20 else 9); e["weakest_kind"] = [wk[0], wk[1]["pred"], wk[1]["base_acc"]]
        # exam-style pass probability at 10%: cells = (src x kind) with 300 locked items each, retention per cell = predicted
        pc = [float(np.mean(v)) for v in cells.values() if len(v) >= 8]
        ok = 0
        for _ in range(2000):
            tot = 0; kept = 0; good = True
            for p in pc:
                nb = 250; k = rng.binomial(nb, p); kept += k; tot += nb
                if wil(k, nb)[1] < 0.95 and k / nb < 0.95: good = False
                if k / nb < 0.95: good = False
            ok += good and kept / tot >= 0.98
        e["sim_pass_prob_r100"] = ok / 2000
        OUT[f"{mod}/{t}"] = e
json.dump(OUT, open(H + "/tour_tables.json", "w"), indent=1)
L = ["| model | task | n | base acc | %bc m>=4 | m>=8 | m>=12 | pred ret 10% | 11.3% | 12.5% | weakest kind (pred, base acc) | sim P(pass 10%) |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
for k, e in OUT.items():
    m, t = k.split("/"); L.append(f"| {m} | {t} | {e['n']} | {e['base_acc']:.3f} | {e['frac_m4']:.3f} | {e['frac_m8']:.3f} | {e['frac_m12']:.3f} | {e['pred_r100']:.3f} | {e['pred_r1125']:.3f} | {e['pred_r125']:.3f} | {e['weakest_kind'][0]} ({e['weakest_kind'][1]:.3f}, {e['weakest_kind'][2]:.2f}) | {e['sim_pass_prob_r100']:.2f} |")
open(H + "/RESULTS_BASE.md", "w").write("\n".join(L) + "\n"); print("\n".join(L))
