"""Scorecard v2 (2026-11-06, after T-REDTEAM): one place that turns raw per-item results into (a) hard rule GATES and (b) soft OPTIMIZATION scores in [0,1].

Why: README says the BEST entry among those passing every rule wins, and the maintainer re-tests on his own OOD data. So we optimise a vector, never one number:
every soft score is mapped 0 = at/below the rule threshold (or worse than base), 0.5 = at the threshold, 1 = target ("excellent"), clipped. Headline = MAXIMIN
(the weakest score) plus the whole vector, so a strong axis can never hide a weak one. Gates are pass/fail and come first; a failed gate => verdict FAIL.

What changed vs v1 (each item is covered by harness/tests/test_scorecard_v2.py and harness/redteam/test_scorecard_and_stats.py):
  * retention >= 95% is gated on EVERY SOURCE and EVERY (source, kind) cell (not only per kind and overall); the Wilson lower bound is reported next to each
    (point estimate gates, as the rule is worded; `strict_wilson` in details shows which cells would also certify with 95% confidence).
  * retention is reported and gated both WITH and WITHOUT S1/nocall (that cell is cheap), and per S1 stratum (seen/unseen) when `stratum[]` is supplied.
  * real - control >= 10 points with a (clustered) CI lower bound > 0 for EVERY control: empty (byte-identical), shuffled / 50-50 mixed-label, random-wrong-tool.
  * no_new_parts is COMPUTED from the artifact tree (audit_artifact_tree: dir + allowed manifest + grep needles). Self-declared extra_files / grep_train_hits are
    ignored (kept only as `declared`); without an `artifact_tree` block the gate fails closed.
  * teach CI = cluster bootstrap (tool family / item cluster x annotation draw, crossed weights) when `cluster` / `draw` arrays are given; plain item bootstrap is
    reported as `item_boot` for comparison and the gate `teach_clusters_given` requires cluster ids.
  * rule 8 is checked per (source, kind) cell: base accuracy must be above chance(+0.15) and below the 0.97 ceiling in every cell.
  * power note: for every retention gate the observed retention needed to certify it with Wilson lo at the ACTUAL n (and whether that is achievable).
  * noise floor (noise_floor.py): `base_runs` = K>=3 original-model runs that differ in batch size / engine. base_correct := intersection (default) or majority of
    the runs; base-vs-base retention is the floor; candidate must be >= 98% AND >= floor - 0.5 pt.
Input (JSON), all per-item arrays aligned by index:
  base_correct[] (ignored if base_runs given), cand_correct[]    0/1 under the task scorer
  base_runs: [{"correct":[...], "engine":"vllm|hf|..", "batch":int}, ...]   K>=3 runs of the original model; base_mode: "intersection"|"majority"
  kind[], source[]  (+ optional stratum[], family[], agree[])
  size: {cand_bytes, base_fp16_bytes}; base_params, generative(bool), kinds_real_output(bool)
  artifact_tree: {dir, allowed_manifest (path|dict|list), needles (list) | needles_file}
  feedback (optional): {n_annotations, bytes_added,
        teach: {n, base_correct[], real_correct[], empty_correct[], shuffled_correct[] or mixed_label_correct[], random_wrong_tool_correct[], cluster[]?, draw[]?},
        neighbor: {n, before_correct[], after_correct[]}, exam_after_correct[] }
  chance: float | {"S1/nocall": x, ...}
  maintainer_ood (optional)
Run:  python scorecard.py results.json   (prints table, writes results.scorecard.json)
"""
import json, sys, math, os, hashlib
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # noise_floor.py lives next to this file

# ------------------------------------------------------------------ statistics
def wilson_lo(k, n, z=1.96):
    if n == 0: return 0.0
    p = k / n; d = 1 + z * z / n
    return float((p + z * z / (2 * n) - z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / d)

def needed_for_wilson(n, thr, z=1.96):
    """smallest k with wilson_lo(k, n) >= thr; None if even k = n cannot certify. Returns (k, k/n)."""
    if n == 0 or wilson_lo(n, n, z) < thr: return None
    lo, hi = 0, n
    while lo < hi:
        m = (lo + hi) // 2
        if wilson_lo(m, n, z) >= thr: hi = m
        else: lo = m + 1
    return lo, lo / n

def lin(x, lo, mid, hi):
    """0 at <=lo, 0.5 at mid, 1 at >=hi (piecewise linear)."""
    if x <= lo: return 0.0
    if x >= hi: return 1.0
    return 0.5 * (x - lo) / (mid - lo) if x < mid else 0.5 + 0.5 * (x - mid) / (hi - mid)

def _boot_means(d, clusters=None, draws=None, B=4000, seed=0):
    """Bootstrap distribution of mean(d). clusters/draws: arrays of group ids. cluster only -> cluster bootstrap; draw only -> draw bootstrap;
    both -> crossed (pigeonhole) bootstrap with weight = w_cluster * w_draw per observation. neither -> item bootstrap."""
    d = np.asarray(d, float); n = len(d); g = np.random.default_rng(seed)
    if n == 0: return np.zeros(1)
    def codes(a): return np.unique(np.asarray(a).astype(str), return_inverse=True)[1]
    if clusters is None and draws is None:
        idx = g.integers(0, n, (B, n)); return d[idx].mean(1)
    cc = codes(clusters) if clusters is not None else None; dd = codes(draws) if draws is not None else None
    out = np.empty(B)
    for b in range(B):
        w = np.ones(n)
        if cc is not None: k = cc.max() + 1; w = w * np.bincount(g.integers(0, k, k), minlength=k)[cc]
        if dd is not None: k = dd.max() + 1; w = w * np.bincount(g.integers(0, k, k), minlength=k)[dd]
        out[b] = (w * d).sum() / w.sum() if w.sum() > 0 else d.mean()
    return out

def paired_boot(d, B=4000, seed=0, clusters=None, draws=None):
    d = np.asarray(d, float)
    if len(d) == 0: return (0.0, 0.0, 0.0)
    m = _boot_means(d, clusters, draws, B, seed)
    return float(d.mean()), float(np.quantile(m, .025)), float(np.quantile(m, .975))

def retention(b, c, mask=None):
    b = np.asarray(b, bool); c = np.asarray(c, bool)
    if mask is not None: m = np.asarray(mask, bool); b, c = b[m], c[m]
    nb = int(b.sum()); k = int((b & c).sum())
    return dict(n_base_correct=nb, kept=k, ret=(k / nb if nb else 0.0), ret_lo=wilson_lo(k, nb))

def cluster_retention_ci(b, c, clusters, B=2000, seed=0):
    """cluster-bootstrap CI of retention = kept / n_base_correct, resampling whole clusters (e.g. API families)."""
    b = np.asarray(b, bool); c = np.asarray(c, bool); cc = np.unique(np.asarray(clusters).astype(str), return_inverse=True)[1]; k = cc.max() + 1
    nb_c = np.bincount(cc, weights=b.astype(float), minlength=k); kp_c = np.bincount(cc, weights=(b & c).astype(float), minlength=k); g = np.random.default_rng(seed)
    r = []
    for _ in range(B):
        w = np.bincount(g.integers(0, k, k), minlength=k); den = (w * nb_c).sum()
        if den > 0: r.append((w * kp_c).sum() / den)
    return float(np.quantile(r, .025)), float(np.quantile(r, .975)), int(k)

# ------------------------------------------------------------------ artifact tree audit (no self-declared fields)
def _sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for ch in iter(lambda: f.read(1 << 24), b""): h.update(ch)
    return h.hexdigest()

def load_manifest(m):
    """allowed manifest: dict {relpath: sha256 | None | "*"} | {"files": {...}} | list of relpaths | path to a JSON file with one of those. None/"*" = any content."""
    if isinstance(m, str): m = json.load(open(m))
    if isinstance(m, dict) and "files" in m: m = m["files"]
    if isinstance(m, (list, tuple, set)): return {p: None for p in m}
    return {k: (None if v in (None, "*") else v) for k, v in m.items()}

def manifest_from_dir(d, pin=("config.json", "tokenizer.json", "tokenizer_config.json", "generation_config.json")):
    """helper for the lock step: allowed manifest of an untouched artifact; weight files may change (None), listed small files are pinned by sha."""
    out = {}
    for root, _, fs in os.walk(d):
        for f in fs:
            p = os.path.join(root, f); rel = os.path.relpath(p, d); out[rel] = _sha_file(p) if os.path.basename(rel) in pin else None
    return out

def audit_artifact_tree(artifact_dir, allowed_manifest, needles=None, needles_file=None, max_needles=500, min_needle_len=12):
    """Computed from the artifact directory itself. extra_files = files present but not in the allowed manifest (symlinks always extra);
    modified_files = pinned-sha mismatches; grep_train_hits = number of (file, needle) pairs where a needle (training / annotation / exam text) occurs verbatim in a file
    (utf-8 and JSON-escaped forms; chunked, so large weight files are scanned too). grep_run=False when no needles were supplied (the gate then fails closed)."""
    man = load_manifest(allowed_manifest); extra, modified, present = [], [], []
    for root, dirs, fs in os.walk(artifact_dir, followlinks=False):
        for f in fs + [d for d in dirs if os.path.islink(os.path.join(root, d))]:
            p = os.path.join(root, f); rel = os.path.relpath(p, artifact_dir); present.append(rel)
            if os.path.islink(p) or rel not in man: extra.append(rel)
            elif man[rel] is not None and _sha_file(p) != man[rel]: modified.append(rel)
    ns = []
    if needles_file:
        raw = open(needles_file).read()
        try: v = json.loads(raw); ns += [str(x) for x in (v if isinstance(v, list) else [v])]
        except Exception: ns += [l.rstrip("\n") for l in raw.splitlines()]
    ns += list(needles or [])
    ns = sorted({n for n in ns if len(n) >= min_needle_len}, key=lambda s: (-len(s), s))[:max_needles]
    forms = []
    for n in ns:
        forms += [(n, n.encode("utf-8"))]
        j = json.dumps(n, ensure_ascii=False)[1:-1].encode("utf-8")
        if j != n.encode("utf-8"): forms.append((n, j))
    hits = []; L = max([len(b) for _, b in forms] + [1])
    if forms:
        for rel in present:
            p = os.path.join(artifact_dir, rel)
            if os.path.islink(p) or not os.path.isfile(p): continue
            found = set(); tail = b""
            with open(p, "rb") as f:
                while True:
                    ch = f.read(1 << 25)
                    if not ch: break
                    buf = tail + ch
                    for n, b in forms:
                        if n not in found and b in buf: found.add(n)
                    tail = buf[-(L - 1):] if L > 1 else b""
            hits += [(rel, n[:60]) for n in sorted(found)]
    return dict(extra_files=sorted(extra), modified_files=sorted(modified), missing_files=sorted(set(man) - set(present)), n_files=len(present),
                grep_run=bool(forms), n_needles=len(ns), grep_train_hits=len(hits), grep_hits=hits[:20])

# ------------------------------------------------------------------ scoring
def score(R):
    kind = np.array(R["kind"]); src = np.array(R["source"]); c = np.array(R["cand_correct"], bool); N = len(c)
    G = {}; S = {}; X = {}; P = {}
    # ---- noise floor / base definition
    runs = R.get("base_runs")
    if runs:
        import noise_floor as NF
        arr = [np.array(r["correct"] if isinstance(r, dict) else r, bool) for r in runs]
        mode = R.get("base_mode", "intersection")
        b = np.all(arr, 0) if mode == "intersection" else (np.mean(arr, 0) >= 0.5)
        fl = NF.floor(arr); X["noise_floor"] = dict(mode=mode, K=len(arr), min_pair=fl["min_pair"], mean_pair=fl["mean_pair"], flip_items=fl["flip_items"], majority_acc=fl["majority_acc"],
                                                    intersection_acc=fl["intersection_acc"], configs=[(r.get("engine"), r.get("batch")) for r in runs if isinstance(r, dict)])
        G["noise_floor_runs>=3"] = len(arr) >= 3 and all(len(a) == N for a in arr)
        cfg = {(r.get("engine"), r.get("batch")) for r in runs if isinstance(r, dict)}
        G["noise_floor_runs_vary_engine_or_batch"] = len(cfg) >= 2 and all(isinstance(r, dict) and r.get("engine") is not None and r.get("batch") is not None for r in runs)
    else:
        b = np.array(R["base_correct"], bool); fl = None
        G["noise_floor_runs>=3"] = False; G["noise_floor_runs_vary_engine_or_batch"] = False
        X["noise_floor"] = "MISSING: supply base_runs (K>=3 original-model runs varying batch size / engine)"
    # ---- size
    ratio = R["size"]["cand_bytes"] / R["size"]["base_fp16_bytes"]; X["size_ratio"] = ratio; G["size<=10%"] = ratio <= 0.10
    # ---- retention: overall, with/without S1 nocall, per source, per kind, per (source, kind) cell, per stratum
    ov = retention(b, c); X["retention_overall"] = ov; G["retention>=98%"] = ov["ret"] >= 0.98
    ex = ~((src == "S1") & (kind == "nocall")); r_ex = retention(b, c, ex); X["retention_excl_S1_nocall"] = r_ex; G["retention_excl_S1_nocall>=98%"] = r_ex["ret"] >= 0.98
    if fl is not None:
        X["floor_comparison"] = dict(cand=ov["ret"], floor_mean_pair=fl["mean_pair"], required=max(0.98, fl["mean_pair"] - 0.005))
        G["retention>=floor-0.5pt"] = ov["ret"] >= fl["mean_pair"] - 0.005
    else: G["retention>=floor-0.5pt"] = False
    per_src = {s: retention(b, c, src == s) | dict(n_locked=int((src == s).sum())) for s in sorted(set(src))}; X["retention_per_source"] = per_src
    G["every_source_retention>=95%"] = all(v["ret"] >= 0.95 for v in per_src.values())
    per = {k: retention(b, c, kind == k) | dict(n_locked=int((kind == k).sum())) for k in sorted(set(kind))}; X["retention_per_kind"] = per
    G["every_kind_retention>=95%"] = all(v["ret"] >= 0.95 for v in per.values())
    cells = {f"{s}/{k}": retention(b, c, (src == s) & (kind == k)) | dict(n_locked=int(((src == s) & (kind == k)).sum())) for s in sorted(set(src)) for k in sorted(set(kind)) if ((src == s) & (kind == k)).any()}
    X["retention_per_cell"] = cells; G["every_source_kind_cell_retention>=95%"] = all(v["ret"] >= 0.95 for v in cells.values())
    if "stratum" in R:
        st = np.array(R["stratum"]); X["retention_per_stratum"] = {s: retention(b, c, st == s) | dict(n_locked=int((st == s).sum())) for s in sorted(set(st))}
    if "family" in R:
        lo_, hi_, nf = cluster_retention_ci(b, c, R["family"]); X["retention_overall_family_cluster_ci"] = dict(lo=lo_, hi=hi_, n_families=nf)
    # ---- exam composition
    G["exam>=2000"] = N >= 2000
    G["every_source_kind_cell>=330"] = all(v["n_locked"] >= 330 for v in cells.values()) and len(cells) >= 1
    ood = src == "S2"; G["ood_block>=500"] = int(ood.sum()) >= 500
    s1k = {k for k in set(kind[src == "S1"])}; s2k = {k for k in set(kind[ood])}; G["ood_block_has_every_S1_kind"] = s1k <= s2k
    # ---- rule 8 per cell
    ch = R.get("chance", 0.0); chd = ch if isinstance(ch, dict) else {}; r8 = {}
    for name in cells:
        s, k = name.split("/"); m = (src == s) & (kind == k); acc = float(b[m].mean()); cc = chd.get(name, chd.get(k, ch if not isinstance(ch, dict) else 0.0))
        r8[name] = dict(base_acc=acc, chance=cc, ok=bool(acc > cc + 0.15 and acc < 0.97))
    X["rule8_per_cell"] = r8; G["base_above_chance_below_ceiling_every_cell"] = all(v["ok"] for v in r8.values())
    G["real_output"] = bool(R.get("kinds_real_output", False))
    # ---- no new parts: computed from the tree
    at = R.get("artifact_tree")
    if at:
        a = audit_artifact_tree(at["dir"], at["allowed_manifest"], at.get("needles"), at.get("needles_file")); X["artifact_tree_audit"] = a
        G["no_new_parts"] = len(a["extra_files"]) == 0 and len(a["modified_files"]) == 0 and a["grep_run"] and a["grep_train_hits"] == 0
    else:
        X["artifact_tree_audit"] = "MISSING: supply artifact_tree {dir, allowed_manifest, needles|needles_file}"; G["no_new_parts"] = False
    if "no_new_parts" in R: X["declared_no_new_parts_IGNORED"] = R["no_new_parts"]
    # ---- feedback
    fb = R.get("feedback")
    if fb:
        t = fb["teach"]; tb, tr = np.array(t["base_correct"], float), np.array(t["real_correct"], float)
        cl, dr = t.get("cluster"), t.get("draw")
        gain, glo, ghi = paired_boot(tr - tb, clusters=cl, draws=dr); ib = paired_boot(tr - tb)
        X["teach_gain"] = dict(mean=gain, lo=glo, hi=ghi, n=t["n"], clustered=cl is not None or dr is not None, n_clusters=(len(set(cl)) if cl is not None else None), n_draws=(len(set(dr)) if dr is not None else None))
        X["teach_gain_item_boot"] = dict(mean=ib[0], lo=ib[1], hi=ib[2])
        ctr = {"empty": t.get("empty_correct"), "shuffled": t.get("shuffled_correct", t.get("mixed_label_correct")), "mixed_label_50_50": t.get("mixed_label_correct"),
               "random_wrong_tool": t.get("random_wrong_tool_correct")}
        sep = {}
        for name, arr in ctr.items():
            if arr is None: continue
            m_, lo_, hi_ = paired_boot(tr - np.array(arr, float), clusters=cl, draws=dr); sep[name] = dict(mean=m_, lo=lo_, hi=hi_); X[f"real_minus_{name}"] = sep[name]
        G["controls_complete(empty,shuffled|mixed,random_wrong_tool)"] = all(k in sep for k in ("empty", "shuffled", "random_wrong_tool"))
        G["controls_clearly_worse(>=10pt,lo>0)"] = bool(sep) and all(v["mean"] >= 0.10 and v["lo"] > 0 for v in sep.values()) and "empty" in sep and "shuffled" in sep
        G["feedback_n<=25"] = fb["n_annotations"] <= 25; G["teach_set>=25"] = t["n"] >= 25
        G["teach_clusters_given"] = cl is not None and len(set(cl)) >= 8
        G["teach_clear_gain(lo>0,mean>=.15)"] = glo > 0 and gain >= 0.15
        nb_ = fb["neighbor"]; nbf, naf = np.array(nb_["before_correct"], bool), np.array(nb_["after_correct"], bool); nr = retention(nbf, naf); X["neighbor_retention"] = nr
        P["neighbor"] = _power("neighbors", nr["n_base_correct"], 0.98)
        G["neighbors_kept>=98%"] = nr["ret"] >= 0.98
        ea = np.array(fb["exam_after_correct"], bool); er = retention(b, ea); X["exam_retention_after_feedback"] = er
        G["exam_after_feedback_retention>=98%"] = er["ret"] >= 0.98
        G["size_after_feedback<=10%"] = (R["size"]["cand_bytes"] + fb.get("bytes_added", 0)) / R["size"]["base_fp16_bytes"] <= 0.10
    else: G["feedback_present"] = False
    # ---- power note
    P["overall"] = _power("overall (98%)", ov["n_base_correct"], 0.98); P["overall_excl_S1_nocall"] = _power("overall excl S1/nocall (98%)", r_ex["n_base_correct"], 0.98)
    for n_, v in per_src.items(): P[f"source:{n_}"] = _power(f"source {n_} (95%)", v["n_base_correct"], 0.95)
    for n_, v in per.items(): P[f"kind:{n_}"] = _power(f"kind {n_} (95%)", v["n_base_correct"], 0.95)
    for n_, v in cells.items(): P[f"cell:{n_}"] = _power(f"cell {n_} (95%)", v["n_base_correct"], 0.95)
    for key, v in P.items():
        v["observed"] = {"overall": ov, "overall_excl_S1_nocall": r_ex, "neighbor": X.get("neighbor_retention")}.get(key) or (per_src.get(key[7:]) if key.startswith("source:") else per.get(key[5:]) if key.startswith("kind:") else cells.get(key[5:]) if key.startswith("cell:") else None)
        v["observed"] = None if v["observed"] is None else v["observed"]["ret"]; v["certified_with_wilson_lo"] = bool(v["observed"] is not None and v["needed_retention"] is not None and v["observed"] >= v["needed_retention"] - 1e-12)
    X["power_note"] = P
    X["strict_wilson"] = {n_: v["certified_with_wilson_lo"] for n_, v in P.items()}
    # ---- soft scores in [0,1]
    S["size_margin"] = 0.0 if ratio > 0.10 else 0.5 + 0.5 * min(1.0, (0.10 - ratio) / 0.005)
    S["retention_overall"] = lin(ov["ret_lo"], 0.95, 0.98, 0.995)
    S["retention_worst_kind"] = lin(min(v["ret_lo"] for v in per.values()), 0.90, 0.95, 0.985)
    S["retention_worst_cell"] = lin(min(v["ret_lo"] for v in cells.values()), 0.90, 0.95, 0.985)
    r1, r2 = retention(b, c, src != "S2"), retention(b, c, ood)
    if r2["n_base_correct"]: X["ood_gap"] = r1["ret"] - r2["ret"]; S["ood_robustness"] = lin(r2["ret_lo"], 0.93, 0.98, 0.995) * (1.0 if X["ood_gap"] <= 0.01 else max(0.0, 1.0 - (X["ood_gap"] - 0.01) / 0.05))
    d = c.astype(float) - b.astype(float); dm, dlo, dhi = paired_boot(d); X["acc_delta_vs_base"] = dict(mean=dm, lo=dlo, hi=dhi)
    harm = float(((b) & (~c)).mean()); help_ = float(((~b) & (c)).mean()); X["harmful_flip_rate"] = harm; X["helpful_flip_rate"] = help_
    S["harmful_flips"] = lin(0.04 - harm, 0.0, 0.02, 0.035)
    if "agree" in R: ag = float(np.mean(R["agree"])); X["behavioural_agreement"] = ag; S["agreement"] = lin(ag, 0.85, 0.95, 0.99)
    if fb:
        S["teach_gain"] = lin(X["teach_gain"]["lo"], 0.0, 0.10, 0.35)
        S["control_separation"] = lin(min(v["lo"] for v in sep.values()), 0.0, 0.10, 0.30) if sep else 0.0
        S["neighbor_retention"] = lin(X["neighbor_retention"]["ret_lo"], 0.93, 0.98, 0.995)
        S["exam_after_feedback"] = lin(X["exam_retention_after_feedback"]["ret_lo"], 0.95, 0.98, 0.995)
        S["feedback_bytes_added"] = 1.0 if fb.get("bytes_added", 0) == 0 else max(0.0, 1.0 - fb["bytes_added"] / 1e5)
    X["difficulty_log10_base_fp16_bytes"] = math.log10(R["size"]["base_fp16_bytes"]); X["generative"] = bool(R.get("generative", False))
    gates_pass = all(G.values()); maximin = min(S.values()) if S else 0.0
    return dict(verdict="PASS_ALL_GATES" if gates_pass else "FAIL", failed_gates=[k for k, v in G.items() if not v], gates=G, scores=S, maximin=maximin,
                weakest=min(S, key=S.get) if S else None, mean_score=float(np.mean(list(S.values()))) if S else 0.0, details=X)

def _power(name, n, thr):
    """observed retention needed for Wilson lo >= thr at the actual number of base-correct items n."""
    r = needed_for_wilson(n, thr)
    return dict(gate=name, n_base_correct=n, threshold=thr, needed_k=None if r is None else r[0], needed_retention=None if r is None else r[1], achievable=r is not None)

def show(o):
    print("VERDICT:", o["verdict"], "| failed:", o["failed_gates"]); print("GATES"); [print(f"  {'ok ' if v else 'XX '}{k}") for k, v in o["gates"].items()]
    print("SCORES (0.5 = rule threshold, 1 = target)"); [print(f"  {v:5.2f}  {k}") for k, v in sorted(o["scores"].items(), key=lambda kv: kv[1])]
    print(f"MAXIMIN {o['maximin']:.2f} (weakest: {o['weakest']})  MEAN {o['mean_score']:.2f}")
    print("POWER NOTE (observed retention needed to certify with Wilson lo at actual n):")
    for k, v in o["details"]["power_note"].items():
        need = "cannot certify at this n" if not v["achievable"] else f"{v['needed_retention']:.4f} ({v['needed_k']}/{v['n_base_correct']})"
        print(f"  {v['gate']:42s} n={v['n_base_correct']:5d} need {need}; observed {v['observed'] if v['observed'] is None else round(v['observed'], 4)} -> {'certified' if v['certified_with_wilson_lo'] else 'NOT certified'}")

if __name__ == "__main__":
    o = score(json.load(open(sys.argv[1]))); show(o); json.dump(o, open(sys.argv[1].replace(".json", ".scorecard.json"), "w"), indent=1, default=float)
