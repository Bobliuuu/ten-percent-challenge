"""usage: python analyze_edit.py results.json [...]  -> markdown table per method x n (draw-level bootstrap CIs)"""
import json, sys, numpy as np, collections
A = lambda s: np.array(list(map(int, s))) if s else np.zeros(0)
def ci(x, B=2000, seed=0):
    x = np.asarray(x, float); r = np.random.default_rng(seed); b = [x[r.integers(0, len(x), len(x))].mean() for _ in range(B)]; return np.percentile(b, [2.5, 97.5])
def an(path):
    R = json.load(open(path)); runs = R["runs"]; meta = R["neigh_meta"]; cells = collections.defaultdict(list)
    for i, (src, g, _) in enumerate(meta): cells[(src, g)].append(i)
    print(f"\n### {path}  model={R['model']}  base teach={R['base']['jb_teach']:.3f}  base labels on annotation pool={R['base_labels']}  base-correct neighbours={len(meta)}")
    print("| method | n | draws | teach real [CI] | shuffled | wrong | mixed | real-shuf [CI] | ood | para | neigh kept | worst draw | min cell(n>=30) | collapsed(<.9) | bytes |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for m in dict.fromkeys(r["m"] for r in runs):
        for n in sorted({r["n"] for r in runs if r["m"] == m}):
            sel = lambda c: sorted([r for r in runs if r["m"] == m and r["n"] == n and r["c"] == c], key=lambda r: r["d"])
            re_ = sel("real")
            if not re_: continue
            t = lambda c, k="teach": np.array([A(r[k]).mean() if len(A(r[k])) else np.nan for r in sel(c)])
            rt, sh = t("real"), t("shuffle")
            nb = np.stack([A(r["neigh"]) for r in re_]); mc = min([nb[:, ix].mean() for ix in cells.values() if len(ix) >= 30] or [np.nan])
            d = rt - sh if len(sh) == len(rt) else np.array([np.nan])
            f = lambda c: f"{np.nanmean(t(c)):.3f}" if len(sel(c)) else "-"
            lo, hi = ci(rt); dl, dh = ci(d) if len(sh) == len(rt) else (np.nan, np.nan)
            print(f"| {m} | {n} | {len(re_)} | {rt.mean():.3f} [{lo:.2f},{hi:.2f}] | {f('shuffle')} | {f('wrong')} | {f('mixed_wrong')} | {d.mean():+.3f} [{dl:+.2f},{dh:+.2f}] | {t('real','ood').mean():.3f} | {t('real','para').mean():.3f} | {nb.mean():.4f} | {nb.mean(1).min():.3f} | {mc:.3f} | {int((nb.mean(1)<0.9).sum())} | {R.get('bytes',{}).get(m,0)} |")
    if R.get("validity"): print("\nreal-generation validity (teacher-forced vs real greedy generation agreement):", [(v['m'][:3], v['set'][:5], round(v['agree'], 3), round(v['acc_gen'], 3)) for v in R["validity"]])
if __name__ == "__main__":
    for p in sys.argv[1:]: an(p)
