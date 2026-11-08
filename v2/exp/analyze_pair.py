"""I3: analysis. Reuses harness/stats.py (McNemar exact, Clopper-Pearson, exact paired non-inferiority bound).
  base   --exp X01 --tag 32b --arm base_32b            : base accuracy per source/class/cell, base-correct counts, floor from pass 1 vs 2, X01 decision rule
  decide --exp E --base ARMTAG --cands a_s11,a_s12 ... : paired retention on SEL ONLY (REP is refused here), CIs, cells, best-of-N winner, MDE
  unseal --exp E --base ARMTAG --cands WINNER[,ALL..] --reason TEXT : appends v2/exp/UNSEAL_LOG.jsonl, then the same tables on REP + tier (section 1.4)
Item files: results/<exp>/<arm>_s<seed>[_p2]_<SET>.items.jsonl (id, src, kind, group, gold, pred, correct, margin)."""
import os, sys, json, math, argparse, random, time, collections
HERE = os.path.dirname(os.path.abspath(__file__)); V2 = os.path.dirname(HERE)
sys.path.insert(0, V2 + "/harness")
import stats as hs
def read(p): return [json.loads(l) for l in open(p)]
def wilson(k, n, z=1.959964):
    if n == 0: return (float("nan"),) * 2
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)); return ((c - h) / d, (c + h) / d)
def items(exp, tag, st): return {r["id"]: r for r in read(f"{HERE}/results/{exp}/{tag}_{st}.items.jsonl")}
def cells(rows, f):   # rows: list of (row, retained_bool) -> dict key -> (k, n, wilson_lb, clopper_lb)
    g = collections.defaultdict(lambda: [0, 0])
    for r, ok in rows: x = g[f(r)]; x[0] += ok; x[1] += 1
    return {str(k): dict(k=v[0], n=v[1], ret=v[0] / v[1], wilson_lb=wilson(v[0], v[1])[0], cp_lb=hs.clopper_pearson(v[0], v[1])[0]) for k, v in sorted(g.items())}
def cluster_boot(rows, B=2000, seed=0):
    gr = collections.defaultdict(lambda: [0, 0])
    for r, ok in rows: gr[r["group"]][0] += ok; gr[r["group"]][1] += 1
    gs = list(gr.values()); rng = random.Random(seed); out = []
    for _ in range(B):
        k = n = 0
        for _ in range(len(gs)): g = gs[rng.randrange(len(gs))]; k += g[0]; n += g[1]
        out.append(k / n)
    out.sort(); return out[int(.025 * B)], out[int(.975 * B)]
def retention(base, cand):
    """base, cand: id->row. retained = cand correct among base-correct."""
    rows = [(b, bool(cand[i]["correct"])) for i, b in base.items() if b["correct"] and i in cand]
    k = sum(ok for _, ok in rows); n = len(rows); lo, hi = wilson(k, n); cl, ch = cluster_boot(rows)
    # discordance for McNemar over ALL items
    bo = sum(1 for i, b in base.items() if b["correct"] and not cand[i]["correct"]); co = sum(1 for i, b in base.items() if not b["correct"] and cand[i]["correct"])
    return dict(n_base_correct=n, retained=k, retention=k / n, flip=1 - k / n, wilson=[lo, hi], cluster=[cl, ch], lb=min(lo, cl),
                by_class=cells(rows, lambda r: r["kind"]), by_source=cells(rows, lambda r: r["src"]), by_cell=cells(rows, lambda r: (r["src"], r["kind"])),
                mcnemar_p=hs.mcnemar_exact(bo, co), b_only=bo, c_only=co), rows
def flip_score(base, cand):   # margin-weighted flip score on SEL (section 1.3)
    return sum(math.exp(-b["margin"] / 2) for i, b in base.items() if b["correct"] and not cand[i]["correct"])
def tier(r):
    cls = [v for v in r["by_class"].values()]; src = [v for v in r["by_source"].values()]
    gated = [v for v in r["by_cell"].values() if v["n"] >= 150]
    wm = min([v["ret"] for v in cls + src] or [0]); wlb = min([v["wilson_lb"] for v in cls + src] or [0]); wc = min([v["ret"] for v in gated] or [0])
    t1 = r["lb"] >= .98 and wlb >= .95 and wc >= .95
    t2 = r["retention"] >= .975 and wm >= .95
    return dict(tier="TIER-1" if t1 else "TIER-2" if t2 else "below-TIER-2", worst_marginal=wm, worst_marginal_lb=wlb, worst_gated_cell=wc, note="byte, floor and TIER-3 (<0.965 @32B, <0.975 @72B) checks are applied by the caller")
def mde(n_pairs_draws=3, draw_sd=0.01, item_se=0.004): return 1.96 * math.sqrt(2 * draw_sd ** 2 / n_pairs_draws + item_se ** 2)

def cmd_base(a):
    r = {}; res = dict(exp=a.exp, tag=a.tag, sets={}, rule8={})
    for st in a.sets.split(","):
        try: p1 = items(a.exp, a.arm, st); p2 = items(a.exp, a.arm + "_p2", st)
        except FileNotFoundError: continue
        if st == "REP" and not a.include_rep_counts: pass   # counts only (no per-item look at predictions)
        sc = {i: x for i, x in p1.items() if x["correct"] is not None}
        n = len(sc); k = sum(x["correct"] for x in sc.values())
        ent = dict(n=n, acc=k / n if n else None, base_correct=k)
        if st in ("SEL", "REP", "INJ"):
            bc = [(x, True) for x in sc.values() if x["correct"]]
            ent["base_correct_by_cell"] = {c: v["n"] for c, v in cells(bc, lambda r: (r["src"], r["kind"])).items()}
            ent["base_correct_by_source"] = {c: v["n"] for c, v in cells(bc, lambda r: r["src"]).items()}
            ent["base_correct_by_class"] = {c: v["n"] for c, v in cells(bc, lambda r: r["kind"]).items()}
            g = collections.defaultdict(lambda: [0, 0])
            for x in sc.values(): g[x["src"]][0] += x["correct"]; g[x["src"]][1] += 1
            ent["acc_by_source"] = {s: v[0] / v[1] for s, v in g.items()}; ent["format_fail"] = sum(1 for x in sc.values() if x["pred"] is None)
            # floor: base-correct in pass 1 that are wrong in pass 2 (and the reverse)
            f12 = sum(1 for i, x in sc.items() if x["correct"] and not p2[i]["correct"]); f21 = sum(1 for i, x in sc.items() if not x["correct"] and p2[i]["correct"])
            pred_diff = sum(1 for i, x in sc.items() if x["pred"] != p2[i]["pred"])
            ent["floor"] = dict(flip_p1_to_p2=f12 / k if k else None, flip_p2_to_p1=f21, n_pred_changed=pred_diff, wilson_ub=1 - wilson(k - f12, k)[0] if k else None)
        elif st == "GEN":
            for role in ("mmlu", "arc"):
                rr = [x for x in sc.values() if x["src"] == role]; ent[role] = sum(x["correct"] for x in rr) / max(1, len(rr))
            ch = [i for i, x in p1.items() if x["src"] == "chat"]; ent["chat_pass_agreement"] = sum(p1[i]["out"] == p2[i]["out"] for i in ch) / max(1, len(ch))
            tk = [x for x in p1.values() if x["src"] == "tokens"]
            if tk: ent["ppl"] = math.exp(-tk[0]["logprob_sum"] / tk[0]["n_tok"])
        res["sets"][st] = ent
    # X01 decision rule
    sel, rep = res["sets"].get("SEL", {}), res["sets"].get("REP", {})
    chk = {}
    if sel:
        chk["SEL_floor_le_0.8pt"] = sel["floor"]["flip_p1_to_p2"] is not None and sel["floor"]["flip_p1_to_p2"] <= 0.008
        chk["SEL_base_correct_ge_3000"] = sel["base_correct"] >= 3000
    if rep:
        chk["REP_base_correct_ge_3000"] = rep["base_correct"] >= 3000
        nonc = sum(v for k2, v in rep["base_correct_by_source"].items() if k2 in ("bbc_news", "huffpost")); chk["REP_unseen_source_base_correct_ge_3000"] = nonc >= 3000; chk["REP_unseen_source_base_correct"] = nonc
        chk["REP_floor_le_0.8pt"] = rep["floor"]["flip_p1_to_p2"] is not None and rep["floor"]["flip_p1_to_p2"] <= 0.008
    for st in ("SEL", "REP"):
        if st in res["sets"]: chk[f"{st}_rule8_acc_by_source_in_(0.60,0.95)"] = {s: (0.60 < v < 0.95) for s, v in res["sets"][st]["acc_by_source"].items()}
    res["x01_checks"] = chk
    flat = [x for v in chk.values() for x in (v.values() if isinstance(v, dict) else [v]) if isinstance(x, bool)]
    res["x01_pass_strict"] = all(flat)
    out = f"{HERE}/results/{a.exp}/base_summary_{a.tag}.json"; json.dump(res, open(out, "w"), indent=1); print(json.dumps(res, indent=1)); print("->", out)

def load_arm(exp, tag, st): return items(exp, tag, st)
def load_base(a, st): return items(a.base_exp or a.exp, a.base, st)
def cmd_decide(a, st="SEL"):
    assert st != "REP" or a.unsealed, "REP is sealed; use --stage unseal"
    base = load_base(a, st); out = dict(exp=a.exp, set=st, base=a.base, arms={})
    scores = {}
    for c in a.cands.split(","):
        cand = load_arm(a.exp, c, st); r, _ = retention(base, cand); r["tier_eval"] = tier(r); r["flip_score"] = flip_score(base, cand); scores[c] = r["flip_score"]; out["arms"][c] = r
    if len(scores) > 1:
        w = min(scores, key=lambda c: (scores[c], out["arms"][c]["flip"])); out["best_of_N_winner"] = w
        out["mean_over_all_N"] = sum(v["retention"] for v in out["arms"].values()) / len(scores); out["min_max_retention"] = [min(v["retention"] for v in out["arms"].values()), max(v["retention"] for v in out["arms"].values())]
    out["mde_3_draws_pt"] = 100 * mde(len(scores) or 1)
    if a.floor_tag:
        fl = load_arm(a.exp, a.floor_tag, st); rf, _ = retention(base, fl); out["floor_retention"] = rf["retention"]
    print(json.dumps(out, indent=1, default=str)); return out
def cmd_unseal(a):
    with open(HERE + "/UNSEAL_LOG.jsonl", "a") as f: f.write(json.dumps(dict(ts=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), exp=a.exp, base=a.base, cands=a.cands, reason=a.reason)) + "\n")
    a.unsealed = True; return cmd_decide(a, "REP")

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("stage", choices=["base", "decide", "unseal"]); ap.add_argument("--exp", default="X01"); ap.add_argument("--tag"); ap.add_argument("--arm")
    ap.add_argument("--sets", default="SEL,REP,GEN,INJ"); ap.add_argument("--base"); ap.add_argument("--cands"); ap.add_argument("--floor-tag"); ap.add_argument("--base-exp"); ap.add_argument("--reason")
    ap.add_argument("--include-rep-counts", action="store_true"); a = ap.parse_args(); a.unsealed = False
    if a.stage == "unseal": assert a.reason, "--reason required"
    {"base": cmd_base, "decide": cmd_decide, "unseal": cmd_unseal}[a.stage](a)
if __name__ == "__main__": main()
