"""Rescore saved generations with the answer-command parser (strip optional 'Category:' prefix), retention on base-correct + per-cell/Wilson.
usage: python rescore.py DS SET RESDIR   (RESDIR has outs_{DS}_{name}_{set}.json and base_{DS}_{set}_1.json; optional {DS}_{name}.json for bytes)"""
import sys, json, re, glob, os, math, collections
def wilson(k, n, z=1.96):
    if n == 0: return 0.0
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); r = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)); return (c - r) / d
norm = lambda t: re.sub(r"^\s*category\s*:\s*", "", (t or "").strip(), flags=re.I).strip()
def main(ds, st, d):
    here = os.path.dirname(os.path.abspath(__file__)); rows = {}
    for l in open(f"{here}/data/{ds}/{st}.jsonl"): r = json.loads(l); rows[r["id"]] = r
    b = json.load(open(f"{d}/base_{ds}_{st}_1.json")); ids = b["ids"]; bc = [norm(o) == rows[i]["answer"] for o, i in zip(b["outs"], ids)]
    out = []
    for f in sorted(glob.glob(f"{d}/outs_{ds}_*_{st}.json")):
        name = os.path.basename(f)[len(f"outs_{ds}_"):-len(f"_{st}.json")]
        if name == "base": continue
        o = json.load(open(f)); assert o["ids"] == ids
        ok = [norm(x) == rows[i]["answer"] for x, i in zip(o["outs"], ids)]
        nb = sum(bc); kk = sum(1 for x, y in zip(ok, bc) if x and y); cells = collections.defaultdict(lambda: [0, 0])
        for x, y, i in zip(ok, bc, ids):
            if y: c = cells[rows[i]["kind"]]; c[0] += 1; c[1] += int(x)
        mc = min(v[1] / v[0] for v in cells.values()); wc = min(wilson(v[1], v[0]) for v in cells.values())
        meta = f"{d}/{ds}_{name}.json"; size = json.load(open(meta))["bytes"]["ratio"] if os.path.exists(meta) else None
        out.append(dict(name=name, size=size, acc=sum(ok) / len(ok), base_acc=nb / len(ids), n_bc=nb, ret=kk / nb, ret_lo=wilson(kk, nb), min_cell=mc, min_cell_lo=wc, cells={k: round(v[1] / v[0], 3) for k, v in sorted(cells.items())}))
    for r in out: print(f"{r['name']:18s} size {100*(r['size'] or 0):5.2f}%  acc {r['acc']:.4f} (base {r['base_acc']:.4f})  ret {r['ret']:.4f} lo {r['ret_lo']:.4f}  mincell {r['min_cell']:.3f} (lo {r['min_cell_lo']:.3f})")
    json.dump(out, open(f"{d}/rescored_{ds}_{st}.json", "w"), indent=1)
if __name__ == "__main__": main(*sys.argv[1:4])
