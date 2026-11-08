"""X13 analysis: retention of base-correct items. usage: analyze.py BASE.jsonl CAND.jsonl [--ids ids.txt] [--set SEL]"""
import json, sys, math, random, collections, argparse
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from x13 import correct
def wilson(k, n, z=1.96):
    if n == 0: return (0, 0)
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)
def load(p, st=None): return {r["id"]: r for r in map(json.loads, open(p)) if st is None or r["set"] == st}
def analyze(bp, cp, st=None, ids=None, boots=2000):
    B, C = load(bp, st), load(cp, st); rows = []
    for i, b in B.items():
        if i not in C or (ids is not None and i not in ids): continue
        if correct(b["out"], b["gold"]): rows.append((i, correct(C[i]["out"], C[i]["gold"]), b["kind"], b["src"], b.get("group") or i))
    n = len(rows); k = sum(r[1] for r in rows); res = dict(n_base_correct=n, retention=k / max(1, n), wilson=wilson(k, n))
    for nm, j in (("kind", 2), ("src", 3)):
        d = collections.defaultdict(list)
        for r in rows: d[r[j]].append(r[1])
        res["by_" + nm] = {x: (sum(v) / len(v), len(v), wilson(sum(v), len(v))[0]) for x, v in sorted(d.items())}
    d = collections.defaultdict(list)
    for r in rows: d[(r[3], r[2])].append(r[1])
    res["by_cell"] = {f"{a}|{b}": (sum(v) / len(v), len(v)) for (a, b), v in sorted(d.items())}
    g = collections.defaultdict(list)
    for r in rows: g[r[4]].append(r[1])
    gl = list(g.values()); rng = random.Random(0); bs = []
    for _ in range(boots):
        s = [gl[rng.randrange(len(gl))] for _ in gl]; t = sum(len(x) for x in s); bs.append(sum(sum(x) for x in s) / t)
    bs.sort(); res["cluster_lb"] = bs[int(0.025 * boots)]
    res["base_acc"] = sum(correct(b["out"], b["gold"]) for i, b in B.items() if ids is None or i in ids) / max(1, len(B))
    return res
if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("base"); ap.add_argument("cand"); ap.add_argument("--set"); ap.add_argument("--ids")
    a = ap.parse_args(); ids = set(open(a.ids).read().split()) if a.ids else None
    print(json.dumps(analyze(a.base, a.cand, a.set, ids), indent=1))
