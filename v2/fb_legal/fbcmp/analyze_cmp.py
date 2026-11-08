import json, sys, numpy as np, collections
A = lambda s: np.array(list(map(int, s)))
def an(path):
    R = json.load(open(path)); runs = R["runs"]; meta = R["neigh_meta"]; tg = R.get("teach_gold")
    cells = collections.defaultdict(list)
    for i, (src, g, _) in enumerate(meta): cells[(src, g)].append(i)
    base = R["base"]["jb_teach"]; print("==", path, "base teach", round(base, 3), "natural", R["base_labels"], "rstat_bytes", R.get("rstat_bytes"), "calib_txt_bytes", R.get("cal_txt_bytes"))
    print("%-20s %3s | real  shuf  wrong mixed | ood  | neigh  worst  mincell ncollapse(<.5) | nrows"%("method","n"))
    rows = []
    for m in dict.fromkeys(r["m"] for r in runs):
        for n in sorted({r["n"] for r in runs if r["m"] == m}):
            sel = lambda c: [r for r in runs if r["m"] == m and r["n"] == n and r["c"] == c]
            re = sel("real"); t = lambda c, k="teach": np.mean([A(r[k]).mean() for r in sel(c)])
            nb = np.stack([A(r["neigh"]) for r in re]); mc = min(nb[:, ix].mean() for ix in cells.values() if len(ix) >= 30)
            x = dict(m=m, n=n, real=t("real"), shuf=t("shuffle"), wrong=t("wrong"), mixed=t("mixed_wrong"), ood=t("real", "ood"), neigh=nb.mean(), worst=nb.mean(1).min(), mincell=mc, coll=int((nb.mean(1) < .5).sum()), nrows=np.mean([r["nrows"] for r in re]))
            if tg and len(set(tg)) > 1:
                for y in sorted(set(tg)):
                    ix = [i for i, g in enumerate(tg) if g == y]; x["real_" + y] = np.mean([A(r["teach"])[ix].mean() for r in re]); x["shuf_" + y] = np.mean([A(r["teach"])[ix].mean() for r in sel("shuffle")])
            rows.append(x)
            print("%-20s %3d | %.3f %.3f %.3f %.3f | %.3f | %.4f %.4f %.3f %d | %.1f" % (m, n, x["real"], x["shuf"], x["wrong"], x["mixed"], x["ood"], x["neigh"], x["worst"], x["mincell"], x["coll"], x["nrows"]) + "".join(" %s=%.2f/%.2f" % (y, x["real_" + y], x["shuf_" + y]) for y in sorted(set(tg)) if tg and len(set(tg)) > 1))
    return rows
if __name__ == "__main__":
    for p in sys.argv[1:]: an(p)
