"""TOOLSEL new-tool feedback aggregation. python analyze.py > table.md
Cluster bootstrap: resample scenarios (new-tool pairs) with replacement, then draws within each scenario; statistic = mean over teach items (teach set fixed per scenario).
Gate G2 (feedback_v3 section 1): gain vs empty >= +15 pts (CI lo > 0); real minus EACH control (wrong-label, 50/50 mixed, derangement) >= +10 pts (CI lo > 0); neighbours N1 and N2 >= 98% (point)."""
import json, glob, numpy as np, math, collections
g = np.random.default_rng(0)
def cboot(M, B=1500):
    """M: list (per scenario) of arrays (draws,) -> grand mean, lo, hi"""
    S = len(M); out = np.empty(B)
    for b in range(B):
        si = g.integers(0, S, S); out[b] = np.mean([M[s][g.integers(0, len(M[s]), len(M[s]))].mean() for s in si])
    return float(np.mean([m.mean() for m in M])), float(np.quantile(out, .025)), float(np.quantile(out, .975))
def wilson_lo(p, n, z=1.96):
    if n == 0: return 0.0
    d = 1 + z * z / n; return (p + z * z / (2 * n) - z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / d
print("| model | arm | method | n ann | scen | draws/scen | base teach | teach real | gain (lo) | real-wrong (lo) | real-mixed (lo) | real-derange (lo) | N1 (lo) | N2 (lo) | G2 |\n|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
summ = collections.defaultdict(list)
for f in sorted(glob.glob("results/*_toolsel_fb.json")):
    R = json.load(open(f)); mdl = R["model"].split("/")[-1]
    for arm in ("N", "D"):
        sc = [s for s in R["scen"] if s.get("arm") == arm and not s.get("skipped") and "draws" in s]
        if not sc: continue
        for method in ["lmhead_lda", "lmhead_plain", "lmhead_plain_xs", "lmhead_cal", "lmhead_cal2"]:
            for n in (3, 5, 8, 12, 25):
                per = {k: [] for k in ("real", "wrong", "mixed", "derange", "N1", "N2")}; bt = []
                for s in sc:
                    D = [d for d in s["draws"] if d["method"] == method and d["n"] == n]
                    if not D: continue
                    bt.append(s["base_teach"])
                    for k in ("real", "wrong", "mixed", "derange"): per[k].append(np.array([d[k]["teach"] for d in D]))
                    per["N1"].append(np.array([d["real"]["N1"] for d in D])); per["N2"].append(np.array([d["real"]["N2"] for d in D]))
                if not bt: continue
                bt_ = np.array(bt); gain = cboot([p - b for p, b in zip(per["real"], bt_)])
                cs = {k: cboot([a - b for a, b in zip(per["real"], per[k])]) for k in ("wrong", "mixed", "derange")}
                n1 = cboot(per["N1"]); n2 = cboot(per["N2"]); realm = cboot(per["real"])
                ok = gain[1] > 0 and gain[0] >= .15 and all(cs[k][1] > 0 and cs[k][0] >= .10 for k in cs) and n1[0] >= .98 and n2[0] >= .98
                summ[(mdl, arm, method)].append((n, ok))
                nn1 = 400 * len(per["N1"][0]) * len(per["N1"]); nn2 = 300 * len(per["N2"][0]) * len(per["N2"])
                print(f"| {mdl} | {arm} | {method} | {n} | {len(bt)} | {len(per['real'][0])} | {bt_.mean():.2f} | {realm[0]:.2f} | {gain[0]:+.2f} ({gain[1]:+.2f}) | {cs['wrong'][0]:+.2f} ({cs['wrong'][1]:+.2f}) | {cs['mixed'][0]:+.2f} ({cs['mixed'][1]:+.2f}) | {cs['derange'][0]:+.2f} ({cs['derange'][1]:+.2f}) | {n1[0]:.3f} ({n1[1]:.3f}) | {n2[0]:.3f} ({n2[1]:.3f}) | {'PASS' if ok else 'FAIL'} |")
print("\n### Passing annotation counts per (model, arm, method)\n| model | arm | method | n values that pass G2 |\n|---|---|---|---|")
for k, v in sorted(summ.items()): print(f"| {k[0]} | {k[1]} | {k[2]} | {[n for n, ok in v if ok] or 'none'} |")
print("\n### scenario info\n| scen | names | n_teach (strict J<0.5) | teach/tool | ann pool | N1 | N2 | base teach | V1 in-menu base acc | lda alpha (dev) | verify teach fast/real (agree) | first tokens |\n|---|---|---|---|---|---|---|---|---|---|---|---|")
for f in sorted(glob.glob("results/*_toolsel_fb.json")):
    R = json.load(open(f))
    for s in R["scen"]:
        v = s.get("verify") or {}
        print(f"| {s['scen']} | {list(s['names'].values())} | {s.get('n_teach')} ({s.get('n_teach_strict')}) | {s.get('teach_per_tool')} | {s.get('n_ann_pool')} | {s.get('n_N1')} | {s.get('n_N2')} | {s.get('base_teach')} | {s.get('base_V1_in_menu_teach_acc')} | {s.get('lda_alpha_chosen_on_dev')} | {v.get('teach_fast')}/{v.get('teach_real')} ({v.get('agree_teach')}) | {s.get('first_token_new')} |")
print("\n### teach accuracy on the STRICT subset (teach queries with word-Jaccard < 0.5 to every annotation-pool query of the same tool) vs all teach, real labels\n| model | arm | method | n ann | all teach | strict teach | #strict items (summed over scen) |\n|---|---|---|---|---|---|---|")
for f in sorted(glob.glob("results/*_toolsel_fb.json")):
    R = json.load(open(f)); mdl = R["model"].split("/")[-1]
    for arm in ("N", "D"):
        sc = [s for s in R["scen"] if s.get("arm") == arm and "draws" in s and s.get("n_teach_strict", 0) > 0]
        for method in ["lmhead_lda", "lmhead_cal"]:
            for n in (5, 25):
                a, b, k = [], [], 0
                for s in sc:
                    idx = s["teach_strict_idx"]
                    for d in s["draws"]:
                        if d["method"] == method and d["n"] == n:
                            v = np.array(d["real"]["teach_items"]); a.append(v.mean()); b.append(v[idx].mean())
                    k += len(idx)
                if a: print(f"| {mdl} | {arm} | {method} | {n} | {np.mean(a):.2f} | {np.mean(b):.2f} | {k} |")
