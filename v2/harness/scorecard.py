"""Scorecard: one place that turns raw per-item results into (a) hard rule GATES and (b) soft OPTIMIZATION scores in [0,1].

Why: README says the BEST entry among those passing every rule wins, and the maintainer re-tests on his own OOD data. So we
optimise a vector, never one number: every soft score is mapped  0 = at/below the rule threshold (or worse than base),
0.5 = at the threshold, 1 = target ("excellent"), clipped. Headline = MAXIMIN (the weakest score) plus the whole vector, so a
strong axis can never hide a weak one. Gates are pass/fail and come first; a failed gate => verdict FAIL regardless of scores.

Input (JSON), all per-item arrays aligned by index:
  base_correct[], cand_correct[]          0/1 (exact-match / label-correct under the task scorer)
  kind[]                                  str: answer class / source / category (rule 2 and 3 "kind")
  source[]                                str: "S1" | "S2" (locked never-touched source) | "S3" ...
  agree[]  (optional)                     0/1 candidate output identical to base output (behavioural fidelity)
  size: {cand_bytes, base_fp16_bytes}
  base_params, generative(bool), kinds_real_output(bool)
  feedback (optional): {n_annotations, bytes_added,
        teach: {n, base_correct[], real_correct[], shuffled_correct[], empty_correct[]},
        neighbor: {n, before_correct[], after_correct[]},
        exam_after_correct[] }      # locked exam after feedback, aligned with the exam arrays
  chance (float), base_acc ceiling check done from base_correct
  no_new_parts: {extra_files: [], grep_train_hits: 0}
  maintainer_ood (optional): same shape as the exam arrays, from S3/own data, for the 'ood_gap' score.
Run:  python scorecard.py results.json   (prints table, writes results.scorecard.json)
"""
import json, sys, math
import numpy as np

def wilson_lo(k, n, z=1.96):
    if n == 0: return 0.0
    p = k / n; d = 1 + z * z / n
    return float((p + z * z / (2 * n) - z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / d)

def lin(x, lo, mid, hi):
    """0 at <=lo, 0.5 at mid, 1 at >=hi (piecewise linear)."""
    if x <= lo: return 0.0
    if x >= hi: return 1.0
    return 0.5 * (x - lo) / (mid - lo) if x < mid else 0.5 + 0.5 * (x - mid) / (hi - mid)

def paired_boot(d, B=4000, seed=0):
    d = np.asarray(d, float); g = np.random.default_rng(seed)
    if len(d) == 0: return (0.0, 0.0, 0.0)
    m = np.array([d[g.integers(0, len(d), len(d))].mean() for _ in range(B)])
    return float(d.mean()), float(np.quantile(m, .025)), float(np.quantile(m, .975))

def retention(b, c, mask=None):
    b = np.asarray(b, bool); c = np.asarray(c, bool)
    if mask is not None: m = np.asarray(mask, bool); b, c = b[m], c[m]
    nb = int(b.sum()); k = int((b & c).sum())
    return dict(n_base_correct=nb, kept=k, ret=(k / nb if nb else 0.0), ret_lo=wilson_lo(k, nb))

def score(R):
    b = np.array(R["base_correct"], bool); c = np.array(R["cand_correct"], bool); kind = np.array(R["kind"]); src = np.array(R["source"])
    N = len(b); G = {}; S = {}; X = {}
    # ---------------- gates (pass/fail, README "How entries are judged")
    ratio = R["size"]["cand_bytes"] / R["size"]["base_fp16_bytes"]; X["size_ratio"] = ratio
    G["size<=10%"] = ratio <= 0.10
    ov = retention(b, c); X["retention_overall"] = ov
    G["retention>=98%"] = ov["ret"] >= 0.98
    per = {}
    for k in sorted(set(kind)): per[k] = retention(b, c, kind == k) | dict(n_locked=int((kind == k).sum()))
    X["retention_per_kind"] = per
    G["every_kind_retention>=95%"] = all(v["ret"] >= 0.95 for v in per.values())
    ood = src == "S2"; G["exam>=2000"] = N >= 2000
    G["every_kind>=300"] = all(v["n_locked"] >= 300 for v in per.values()); G["ood_block>=500"] = int(ood.sum()) >= 500
    ch = R.get("chance", 0.0); G["base_above_chance_below_ceiling"] = (b.mean() > ch + 0.15) and (b.mean() < 0.97)
    G["real_output"] = bool(R.get("kinds_real_output", False))
    npn = R.get("no_new_parts", {}); G["no_new_parts"] = (len(npn.get("extra_files", [])) == 0) and npn.get("grep_train_hits", 1) == 0
    fb = R.get("feedback")
    if fb:
        t = fb["teach"]; tb, tr = np.array(t["base_correct"], float), np.array(t["real_correct"], float)
        gain, glo, ghi = paired_boot(tr - tb); X["teach_gain"] = dict(mean=gain, lo=glo, hi=ghi, n=t["n"])
        sh = paired_boot(np.array(t["real_correct"], float) - np.array(t["shuffled_correct"], float)); em = paired_boot(np.array(t["real_correct"], float) - np.array(t["empty_correct"], float))
        X["real_minus_shuffled"] = dict(mean=sh[0], lo=sh[1]); X["real_minus_empty"] = dict(mean=em[0], lo=em[1])
        G["feedback_n<=25"] = fb["n_annotations"] <= 25; G["teach_set>=25"] = t["n"] >= 25
        G["teach_clear_gain(lo>0,mean>=.15)"] = glo > 0 and gain >= 0.15
        G["controls_clearly_worse(lo>0)"] = sh[1] > 0 and em[1] > 0
        nb_ = fb["neighbor"]; nbf, naf = np.array(nb_["before_correct"], bool), np.array(nb_["after_correct"], bool); nr = retention(nbf, naf); X["neighbor_retention"] = nr
        G["neighbors_kept>=98%"] = nr["ret"] >= 0.98
        ea = np.array(fb["exam_after_correct"], bool); er = retention(b, ea); X["exam_retention_after_feedback"] = er
        G["exam_after_feedback_retention>=98%"] = er["ret"] >= 0.98
        G["size_after_feedback<=10%"] = (R["size"]["cand_bytes"] + fb.get("bytes_added", 0)) / R["size"]["base_fp16_bytes"] <= 0.10
    else:
        G["feedback_present"] = False
    # ---------------- soft scores in [0,1]  (0.5 = rule threshold, 1 = target)
    S["size_margin"] = 0.0 if ratio > 0.10 else 0.5 + 0.5 * min(1.0, (0.10 - ratio) / 0.005)  # 0.5 at 10%, 1.0 at <=9.5%
    S["retention_overall"] = lin(ov["ret_lo"], 0.95, 0.98, 0.995)
    S["retention_worst_kind"] = lin(min(v["ret_lo"] for v in per.values()), 0.90, 0.95, 0.985)
    s1 = src != "S2"
    r1, r2 = retention(b, c, s1), retention(b, c, ood)
    if r2["n_base_correct"]: X["ood_gap"] = r1["ret"] - r2["ret"]; S["ood_robustness"] = lin(r2["ret_lo"], 0.93, 0.98, 0.995) * (1.0 if X["ood_gap"] <= 0.01 else max(0.0, 1.0 - (X["ood_gap"] - 0.01) / 0.05))
    d = c.astype(float) - b.astype(float); dm, dlo, dhi = paired_boot(d); X["acc_delta_vs_base"] = dict(mean=dm, lo=dlo, hi=dhi)
    harm = float(((b) & (~c)).mean()); help_ = float(((~b) & (c)).mean()); X["harmful_flip_rate"] = harm; X["helpful_flip_rate"] = help_
    S["harmful_flips"] = lin(0.04 - harm, 0.0, 0.02, 0.035)    # <=0.5% harmful flips -> ~1
    if "agree" in R: ag = float(np.mean(R["agree"])); X["behavioural_agreement"] = ag; S["agreement"] = lin(ag, 0.85, 0.95, 0.99)
    if fb:
        S["teach_gain"] = lin(X["teach_gain"]["lo"], 0.0, 0.10, 0.35)
        S["control_separation"] = lin(min(X["real_minus_shuffled"]["lo"], X["real_minus_empty"]["lo"]), 0.0, 0.10, 0.30)
        S["neighbor_retention"] = lin(X["neighbor_retention"]["ret_lo"], 0.93, 0.98, 0.995)
        S["exam_after_feedback"] = lin(X["exam_retention_after_feedback"]["ret_lo"], 0.95, 0.98, 0.995)
        S["feedback_bytes_added"] = 1.0 if fb.get("bytes_added", 0) == 0 else max(0.0, 1.0 - fb["bytes_added"] / 1e5)
    # difficulty/utility (informational, not maximin-gated): log10 base fp16 bytes, generative output, commercial task flag
    X["difficulty_log10_base_fp16_bytes"] = math.log10(R["size"]["base_fp16_bytes"]); X["generative"] = bool(R.get("generative", False))
    S_core = {k: v for k, v in S.items()}
    gates_pass = all(G.values()); maximin = min(S_core.values()) if S_core else 0.0
    out = dict(verdict="PASS_ALL_GATES" if gates_pass else "FAIL", failed_gates=[k for k, v in G.items() if not v], gates=G, scores=S,
               maximin=maximin, weakest=min(S_core, key=S_core.get) if S_core else None, mean_score=float(np.mean(list(S_core.values()))) if S_core else 0.0, details=X)
    return out

def show(o):
    print("VERDICT:", o["verdict"], "| failed:", o["failed_gates"]); print("GATES"); [print(f"  {'ok ' if v else 'XX '}{k}") for k, v in o["gates"].items()]
    print("SCORES (0.5 = rule threshold, 1 = target)"); [print(f"  {v:5.2f}  {k}") for k, v in sorted(o["scores"].items(), key=lambda kv: kv[1])]
    print(f"MAXIMIN {o['maximin']:.2f} (weakest: {o['weakest']})  MEAN {o['mean_score']:.2f}")

if __name__ == "__main__":
    o = score(json.load(open(sys.argv[1]))); show(o); json.dump(o, open(sys.argv[1].replace(".json", ".scorecard.json"), "w"), indent=1, default=float)
