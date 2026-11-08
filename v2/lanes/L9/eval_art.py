"""Evaluate the PACKED artifact: python eval_art.py ART DATA_DIR OUT_JSON [ds]. Retention vs the unquantised base outputs (base_*_1.json from core235), per source|kind cells, Wilson bounds."""
import sys, os, json, math, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pack235 as P
def wilson(k, n, z=1.96):
    if n == 0: return 0.0
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); r = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)); return (c - r) / d
if __name__ == "__main__":
    ART, DD, OUTJ = sys.argv[1:4]; ds = sys.argv[4] if len(sys.argv) > 4 else "skill7"; BASE = os.environ.get("L9_BASEDIR", "/data/out/res")
    m, tok = P.build_model(ART, os.environ.get("L9_GPUMEM", "120GiB")); res = {}
    for sn in os.environ.get("L9_SETS", "SEL,REP").split(","):
        rows = [json.loads(l) for l in open(f"{DD}/{ds}_{sn.lower()}.jsonl")]
        order = sorted(range(len(rows)), key=lambda i: len(rows[i]["user"])); outs = [None] * len(rows)
        for s in range(0, len(order), 64):
            idx = order[s:s + 64]
            for i, o in zip(idx, P.answer_text(m, tok, [rows[i]["user"] for i in idx])): outs[i] = o
        b = json.load(open(f"{BASE}/base_{ds}_{sn}_1.json")); assert b["ids"] == [r["id"] for r in rows]
        corr = [o == r["answer"] for o, r in zip(outs, rows)]; bc = [o == r["answer"] for o, r in zip(b["outs"], rows)]
        nb = sum(bc); kk = sum(1 for x, y in zip(corr, bc) if x and y); cells = collections.defaultdict(lambda: [0, 0])
        for x, y, r in zip(corr, bc, rows):
            if y: c = cells[r["src"] + "|" + r["kind"]]; c[0] += 1; c[1] += int(x)
        res[sn] = dict(n=len(rows), acc=sum(corr) / len(rows), base_acc=nb / len(rows), n_bc=nb, kept_bc=kk, ret=kk / nb, ret_lo=wilson(kk, nb), cells={k: dict(n=v[0], ret=v[1] / v[0], lo=wilson(v[1], v[0])) for k, v in sorted(cells.items())}, min_cell=min(v[1] / v[0] for v in cells.values()), outs=outs)
        print(sn, {k: v for k, v in res[sn].items() if k not in ("cells", "outs")}, flush=True)
    json.dump(dict(res, ids={sn: [json.loads(l)["id"] for l in open(f"{DD}/{ds}_{sn.lower()}.jsonl")] for sn in res}), open(OUTJ, "w"))
