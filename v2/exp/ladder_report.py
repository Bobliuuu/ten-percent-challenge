"""X03 analysis (SEL ONLY; REP is never read here). Produces v2/exp/results/X03/ladder.json and ladder_table.md.
  python ladder_report.py [--stack-json stack.json]
Pre-registered rules (EXPERIMENT_QUEUE X03, fixed before data):
  KEEP  : paired delta vs previous KEPT rung >= +0.5 pt AND CI lower bound > -0.3 pt; or delta in [-0.5,+0.5] and the piece frees bytes (A2).
  DROP  : delta < -0.5 pt, or the 300-item screen failed.   Anything else = NEUTRAL/UNCERTAIN -> not kept (we say so).
  Screen: first 75 base-correct SEL items per class by sha1(id); kill if retention < same-seed reference - 3.0 pt, or < 0.90, or rel proxy err > 1.5x baseline.
  Gate: A_best - A0 >= +2.0 pt and no class marginal below A0's -> continue; < +1.0 -> not worth the complexity.
  MDE is RECOMPUTED from the measured A0 draw sd (8 draws) before any keep/drop."""
import os, sys, json, math, glob, collections, hashlib
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); RD = HERE + "/results/X03"; BASE = HERE + "/results/X01/base_14b_s0"
sys.path.insert(0, HERE); import ladder
def read(p): return [json.loads(l) for l in open(p)]
def load(tag, st="SEL"):
    p = f"{RD}/{tag}_{st}.items.jsonl"; return {r["id"]: r for r in read(p)} if os.path.exists(p) else None
base = {r["id"]: r for r in read(BASE + "_SEL.items.jsonl")}
bc = [i for i, r in base.items() if r["correct"]]; bc_rows = [base[i] for i in bc]
grp_names = sorted({base[i]["group"] for i in bc}); gidx = {g: k for k, g in enumerate(grp_names)}; G = len(grp_names)
item_g = np.array([gidx[base[i]["group"]] for i in bc]); n_g = np.bincount(item_g, minlength=G).astype(float)
screen = set(ladder.screen_ids(list(base.values())))
KINDS = ("World", "Sports", "Business", "Technology"); SRCS = ("ag_news", "bbc_news")
def flips(tag):
    it = load(tag)
    if it is None: return None
    return np.array([0 if it[i]["correct"] else 1 for i in bc], float)
def ret(f, mask=None): f = f if mask is None else f[mask]; return 1 - f.mean()
def seeds_of(arm): return sorted(int(os.path.basename(p).split("_s")[1].split("_")[0]) for p in glob.glob(f"{RD}/{arm}_s*_SEL.items.jsonl") if "_p2" not in p)
def boot_delta(Fa, Fb, B=2000, seed=0):
    """Fa, Fb: dict seed -> flip vector, matched seeds. delta (pt) = retention(b) - retention(a) = mean(flip_a - flip_b); items clustered by group, draws resampled."""
    ss = sorted(set(Fa) & set(Fb)); S = len(ss)
    d = np.zeros((S, G))
    for k, s in enumerate(ss): np.add.at(d[k], item_g, Fa[s] - Fb[s])
    point = d.sum() / S / n_g.sum(); rng = np.random.default_rng(seed); out = np.empty(B)
    for b in range(B):
        gi = rng.integers(0, G, G); si = rng.integers(0, S, S); out[b] = d[np.ix_(si, gi)].sum() / S / n_g[gi].sum()
    lo, hi = np.percentile(out, [2.5, 97.5]); return 100 * point, 100 * lo, 100 * hi, S
def summarize(arm, only=None):
    ss = [x for x in seeds_of(arm) if only is None or x in only]; F = {s: flips(f"{arm}_s{s}") for s in ss}
    if not ss: return None
    r = {s: ret(F[s]) for s in ss}
    cls = {k: np.mean([ret(F[s], np.array([x["kind"] == k for x in bc_rows])) for s in ss]) for k in KINDS}
    src = {k: np.mean([ret(F[s], np.array([x["src"] == k for x in bc_rows])) for s in ss]) for k in SRCS}
    qj = [json.load(open(p)) for p in glob.glob(f"{RD}/{arm}_s*_quant.json")]
    ratio = [q["ratio_pct"] for q in qj if "ratio_pct" in q]; relerr = {q["cfg"]["seed"]: q.get("rel_proxy_err") for q in qj}
    gen = {}
    for s in ss:
        g = load(f"{arm}_s{s}", "GEN")
        if g: gen[s] = gen_metrics(g)
    return dict(arm=arm, seeds=ss, F=F, ret=r, mean=float(np.mean(list(r.values()))), min=min(r.values()), max=max(r.values()), cls=cls, src=src, ratio=ratio, relerr=relerr, gen=gen,
                screen={s: float(1 - np.mean([F[s][k] for k, i in enumerate(bc) if i in screen])) for s in ss})
_bg = None
def base_gen():
    global _bg
    if _bg is None: _bg = {r["id"]: r for r in read(BASE + "_GEN.items.jsonl")}
    return _bg
def gen_metrics(g):
    b = base_gen(); out = {}
    for role in ("mmlu", "arc"):
        ids = [i for i, r in g.items() if r["src"] == role]; out[role] = float(np.mean([g[i]["correct"] for i in ids])) if ids else None
    ch = [i for i, r in b.items() if r["src"] == "chat" and i in g]; out["chat_agree"] = float(np.mean([g[i]["out"] == b[i]["out"] for i in ch])) if ch else None
    tk = [r for r in g.values() if r["src"] == "tokens"]; out["ppl"] = math.exp(-tk[0]["logprob_sum"] / tk[0]["n_tok"]) if tk else None
    return out
def base_gen_summary():
    b = base_gen(); return gen_metrics(b) | dict(chat_agree=1.0)

def main():
    arms = [a for a in ("A0", "A0p", "A1", "A2", "A4", "A5") if seeds_of(a)]; S = {a: summarize(a) for a in arms}; S["A0_s11-13"] = summarize("A0", {11, 12, 13}); out = dict(arms={}, note="SEL only")
    bg = base_gen_summary(); out["base_gen"] = bg
    # --- draw noise (A0, up to 8 draws)
    a0 = S["A0"]; rets = np.array([a0["ret"][s] for s in a0["seeds"]]); sd = float(rets.std(ddof=1)) if len(rets) > 1 else float("nan")
    # item-sampling SE of a paired delta, measured from A0 seed pairs
    pair_se = []
    for i, s in enumerate(a0["seeds"][:3]):
        for t in a0["seeds"][i + 1:3]: d = a0["F"][s] - a0["F"][t]; pair_se.append(float(d.std(ddof=1) / math.sqrt(len(d))))
    item_se = float(np.mean(pair_se)) if pair_se else 0.004
    mde = lambda n: 1.96 * math.sqrt(2 * sd ** 2 / n + item_se ** 2) * 100
    med = float(np.median(rets)); keep_ = rets[np.abs(rets - med) <= 0.05]; sd_r = float(keep_.std(ddof=1)) if len(keep_) > 1 else float("nan")
    out["draw_noise_robust"] = dict(median=med, n_collapse_draws=int(len(rets) - len(keep_)), collapse_rule="|ret - median| > 5 pt", sd_pt_without_collapse=100 * sd_r, mde_3seeds_pt=1.96 * math.sqrt(2 * sd_r ** 2 / 3 + item_se ** 2) * 100)
    out["draw_noise"] = dict(n_draws=len(rets), retentions=[float(x) for x in rets], mean=float(rets.mean()), sd_pt=100 * sd, range_pt=100 * float(rets.max() - rets.min()), item_se_pair_pt=100 * item_se,
                             mde_3seeds_pt=mde(3), mde_5seeds_pt=mde(5), prereg_assumed_sd_pt=1.0, prereg_mde_3seeds_pt=1.6)
    # --- rungs: previous KEPT rung logic (A3 pending; A0p is a bytes control, not a rung)
    order = ["A1", "A2", "A4", "A5"]; kept = []; prev = "A0"; rung = {}
    for arm in order:
        if arm not in S: rung[arm] = dict(status="NOT RUN"); continue
        s = S[arm]; ref = S[prev]; d, lo, hi, n = boot_delta(ref["F"], s["F"])
        scr = {}
        for sd_ in s["seeds"]:
            r0 = S["A0"]; refscr = r0["screen"].get(sd_) if prev == "A0" else S[prev]["screen"].get(sd_)
            scr[sd_] = dict(arm=s["screen"][sd_], ref=refscr, kill=(refscr is not None and s["screen"][sd_] < refscr - 0.03) or s["screen"][sd_] < 0.90 or
                            (s["relerr"].get(sd_) and S["A0"]["relerr"].get(sd_) and s["relerr"][sd_] > 1.5 * S["A0"]["relerr"][sd_]))
        screen_fail = any(v["kill"] for v in scr.values())
        frees = arm == "A2"
        if screen_fail: status = "DROP (screen)"
        elif d < -0.5: status = "DROP"
        elif d >= 0.5 and lo > -0.3: status = "KEEP"
        elif -0.5 <= d <= 0.5 and frees: status = "KEEP (neutral, frees bytes/risk)"
        elif d >= 0.5: status = "NOT KEPT (delta>=+0.5 but CI lower bound %.2f <= -0.3: uncertain)" % lo
        else: status = "NOT KEPT (neutral: |delta|<0.5, piece frees nothing)"
        dA0, loA0, hiA0, _ = boot_delta(a0["F"], s["F"])
        rung[arm] = dict(prev=prev, delta_pt=d, ci=[lo, hi], n_seeds=n, vs_A0_pt=dA0, vs_A0_ci=[loA0, hiA0], status=status, screen=scr, mde_pt=mde(max(n, 1)), bytes_equiv_pp=(d / 1.75) if True else None)
        if status.startswith("KEEP"): kept.append(arm); prev = arm
    out["rungs"] = rung; out["kept"] = kept
    best = kept[-1] if kept else "A0"
    if best != "A0":
        d, lo, hi, n = boot_delta(S["A0_s11-13"]["F"], S[best]["F"]); cm = {k: (S["A0_s11-13"]["cls"][k], S[best]["cls"][k]) for k in KINDS}
        gate = dict(A_best=best, delta_vs_A0_pt=d, ci=[lo, hi], class_marginals_A0_vs_best=cm, class_below_A0=[k for k, (x, y) in cm.items() if y < x],
                    verdict=("CONTINUE to 32B (X04)" if d >= 2.0 and not [k for k, (x, y) in cm.items() if y < x] else "stack not worth its complexity (X04 runs A0 vs A1+A2 only)" if d < 1.0 else "intermediate: +1.0..+2.0 or a class marginal below A0 (rule: no continue)"))
        wscore = {sd_: sum(math.exp(-base[i]["margin"] / 2) for k, i in enumerate(bc) if S[best]["F"][sd_][k] > 0) for sd_ in S[best]["seeds"]}; w = min(wscore, key=wscore.get)
        gate["A7_best_of_N"] = dict(N=len(wscore), winner_seed=w, winner_ret_sel=S[best]["ret"][w], mean_all_N=S[best]["mean"], min=S[best]["min"], max=S[best]["max"], note="N=3 (not 8: budget); winner chosen on SEL by margin-weighted flip score, so its SEL number is optimistic by construction; REP not unsealed unless stated below")
        out["gate"] = gate
    if "A0p" in S:
        d, lo, hi, n = boot_delta(a0["F"], S["A0p"]["F"]); out["A0p_vs_A0"] = dict(delta_pt=d, ci=[lo, hi], n_seeds=n, ratio=S["A0p"]["ratio"])
    for a, s in S.items(): out["arms"][a] = {k: v for k, v in s.items() if k != "F"}
    out["arms"] = {("A0 (3 draws, seeds 11-13)" if k == "A0_s11-13" else "A0 (all 8 draws)" if k == "A0" else k): v for k, v in out["arms"].items()}
    json.dump(out, open(RD + "/ladder.json", "w"), indent=1, default=float); write_md(out, S, mde); print(json.dumps(out["draw_noise"], indent=1)); print(json.dumps({a: (r.get("status"), r.get("delta_pt")) for a, r in rung.items()}, indent=1))

def pct(x): return "n/a" if x is None else f"{100 * x:.2f}"
def write_md(out, S, mde):
    L = ["# X03 ladder (Qwen2.5-14B-Instruct, NEWS, SEL only, REP sealed)", "", "Generated by `v2/exp/ladder_report.py` from `results/X03/*_SEL.items.jsonl`; paired vs base from X01. Retention = P(candidate correct | base correct), real greedy generation. All arms at real 9.90% (A0p at 10.00%) by `exp/bytes.py` on the written pack; evaluated model = output of the shipped `decode_pack.py`.", ""]
    dn = out["draw_noise"]
    L += ["## Draw noise (A0, same recipe, different Hadamard sign seeds)", f"- draws: {dn['n_draws']}; retentions (pt): " + ", ".join(f"{100 * x:.2f}" for x in dn["retentions"]),
          f"- mean {100 * dn['mean']:.2f}, **sd {dn['sd_pt']:.2f} pt** (pre-registered assumption 1.0), range {dn['range_pt']:.2f} pt; paired item-sampling SE {dn['item_se_pair_pt']:.2f} pt",
          f"- robust: median {100 * out['draw_noise_robust']['median']:.2f}; collapse draws (>5 pt from median) {out['draw_noise_robust']['n_collapse_draws']}; sd without them {out['draw_noise_robust']['sd_pt_without_collapse']:.2f} pt -> MDE(3 seeds) {out['draw_noise_robust']['mde_3seeds_pt']:.2f} pt",
          f"- **MDE recomputed**: {dn['mde_3seeds_pt']:.2f} pt with 3 seeds, {dn['mde_5seeds_pt']:.2f} pt with 5 (pre-registered 1.6 pt at sd 1.0)", ""]
    L += ["## Arms (mean over seeds)", "", "| arm | stack | seeds | real ratio % | SEL retention mean [min,max] | per-class (W/S/B/T) | per-source (AG/BBC) | MMLU / ARC | chat agree | ppl |", "|---|---|---|---|---|---|---|---|---|---|"]
    bg = out["base_gen"]; L.append(f"| base | - | - | 100 | 100 | - | - | {pct(bg['mmlu'])} / {pct(bg['arc'])} | 100 | {bg['ppl']:.3f} |")
    for a, s in out["arms"].items():
        g = s["gen"]; mm = np.mean([x["mmlu"] for x in g.values()]) if g else None; ar = np.mean([x["arc"] for x in g.values()]) if g else None; ca = np.mean([x["chat_agree"] for x in g.values()]) if g else None
        pp = np.mean([x["ppl"] for x in g.values() if x["ppl"]]) if g else None
        L.append(f"| {a} | {','.join(out['kept'][:out['kept'].index(a)]) if a in out['kept'] else '-'} | {','.join(map(str, s['seeds']))} | {np.mean(s['ratio']):.3f} | {100 * s['mean']:.2f} [{100 * s['min']:.2f},{100 * s['max']:.2f}] | "
                 f"{'/'.join(pct(s['cls'][k]) for k in ('World', 'Sports', 'Business', 'Technology'))} | {pct(s['src']['ag_news'])}/{pct(s['src']['bbc_news'])} | {pct(mm)} / {pct(ar)} | {pct(ca)} | {'n/a' if pp is None else f'{pp:.3f}'} |")
    L += ["", "## Rungs: paired deltas vs the previous KEPT rung (seed-matched, items clustered by group, draws resampled, 95% CI)", "", "| arm | vs | delta pt [95% CI] | seeds | vs A0 pt [CI] | decision | bytes-equivalent (pp of size) |", "|---|---|---|---|---|---|---|"]
    for a, r in out["rungs"].items():
        if r["status"] == "NOT RUN": L.append(f"| {a} | - | - | 0 | - | NOT RUN | - |"); continue
        L.append(f"| {a} | {r['prev']} | {r['delta_pt']:+.2f} [{r['ci'][0]:+.2f}, {r['ci'][1]:+.2f}] | {r['n_seeds']} | {r['vs_A0_pt']:+.2f} [{r['vs_A0_ci'][0]:+.2f}, {r['vs_A0_ci'][1]:+.2f}] | {r['status']} | {r['bytes_equiv_pp']:+.2f} |")
    if "A0p_vs_A0" in out:
        x = out["A0p_vs_A0"]; L += ["", f"A0+ (A0 at 10.00%, ratio {np.mean(x['ratio']):.3f}%) vs A0: {x['delta_pt']:+.2f} pt [{x['ci'][0]:+.2f}, {x['ci'][1]:+.2f}] over {x['n_seeds']} seeds (what ~0.1 pp of bytes alone buys)."]
    L += ["", "A3 (GPTAQ): PENDING/NOT RUN. X02 did not deliver a passing fix (v2/quality/gptaq_fix.md: fixed residual-form target worse than baseline on 7B, non-harm criterion fails; quality lane: no further spend). A6 (trellis/Leech): NOT RUN. The VQ lane's own 7B evidence (TCQ L12 V2 at 9.99%: INJ 0.940, NEWS 0.931) is not >= its shared-codebook VQ (INJ 0.95-0.96) at equal bytes, so the pre-registered admission condition (TCQ >= VQ at equal bytes) is not met."]
    if "gate" in out:
        g = out["gate"]; L += ["", "## Gate (pre-registered)", f"- A_best = **{g['A_best']}** (cumulative stack of KEPT rungs: {', '.join(out['kept'])}); paired delta vs A0 (seeds 11-13): **{g['delta_vs_A0_pt']:+.2f} pt** [{g['ci'][0]:+.2f}, {g['ci'][1]:+.2f}] (need >= +2.0 and no class marginal below A0)",
              "- class marginals A0 -> A_best: " + "; ".join(f"{k} {100 * x:.1f} -> {100 * y:.1f}" for k, (x, y) in g["class_marginals_A0_vs_best"].items()), f"- classes below A0: {g['class_below_A0'] or 'none'}; verdict: **{g['verdict']}**",
              f"- A7 best-of-N: N={g['A7_best_of_N']['N']} (not 8), winner seed {g['A7_best_of_N']['winner_seed']} SEL {100 * g['A7_best_of_N']['winner_ret_sel']:.2f}; mean of all N {100 * g['A7_best_of_N']['mean_all_N']:.2f} (min {100 * g['A7_best_of_N']['min']:.2f}, max {100 * g['A7_best_of_N']['max']:.2f}); winner picked on SEL by margin-weighted flip score, so optimistic by construction."]
    L += ["", "## Caveats", "- SEL only; REP sealed (see unseal note if any). Retention here is NOT rule-2 evidence: no arm reaches 98% overall or 95% in Technology; the best is a 14B ladder rung, not a ship candidate.",
          "- Screens were computed from the full-SEL per-item files restricted to the 300 fixed ids (identical rule, no separate vLLM launch); all three seeds of each arm were launched concurrently (cost), the s11 screen verdict is reported per seed.",
          "- General capability (GEN) is far below base for EVERY arm (MMLU about 45-53 vs 78.9 base, chat-agreement under 0.5, ppl 19-31 vs 6.1): the 14B at 9.9% is a NEWS-faithful model, not a general one (internal gate MMLU ratio >= 0.90 fails for all arms; not one of the nine rules; flagged for rule 9 / the maintainer's own-data check).",
          "- A0 has one collapse draw out of 8 (seed 18: the Sports label is generated as 'sport'; strict exact-string scoring). Draw noise is heavy-tailed; paired deltas use seeds 11-13 only.",
          "- Keep/drop uses the pre-registered thresholds as written; KEEP at delta >= +0.5 is a point-estimate rule: A1 and A2 have CI lower bounds only just above 0, A4's CI includes 0."]
    open(RD + "/ladder_table.md", "w").write("\n".join(L) + "\n")
if __name__ == "__main__": main()
