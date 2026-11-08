"""analyze.py base.json cand.json [more cand.json ...] : accuracy of each, and retention of each cand vs base (P(cand ok | base ok)), per (task,source,kind) cells."""
import sys, json, os, collections, math
V2 = os.path.expanduser("~/ten-percent/v2")
sys.path.insert(0, V2 + "/tournament/T1"); sys.path.insert(0, V2 + "/tournament/TOX"); sys.path.insert(0, V2 + "/tournament/INJ")
import importlib.util
def load(p, n):
    s = importlib.util.spec_from_file_location(n, p); m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
T1 = load(V2 + "/tournament/T1/scorer.py", "t1s"); TOX = load(V2 + "/tournament/TOX/scorer.py", "toxs"); INJ = load(V2 + "/tournament/INJ/scorer.py", "injs")
items = {json.loads(l)["id"]: json.loads(l) for l in open(V2 + "/tournament/T1/probe_items.jsonl")}
def ok_rows(R):
    rows = {}
    for r in R["t1"]:
        it = items[r["id"]]; txt = r["out"].split("<|im_end|>")[0]
        try: ok = bool(T1.score_item(txt, it)["correct"])
        except Exception: ok = False
        rows[("t1", r["id"])] = (ok, ("T1", r["source"], r["kind"]))
    for r in R["cls"]:
        ok = (TOX.correct(r["out"], r["gold"]) if r["task"] == "tox" else INJ.correct(r["out"], r["gold"]))
        rows[(r["task"], r["id"])] = (ok, (r["task"].upper(), r.get("source"), r["gold"]))
    return rows
def wilson_lo(k, n, z=1.96):
    if n == 0: return 0.0
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)); return (c - h) / d
def main():
    B = ok_rows(json.load(open(sys.argv[1]))); out = {}
    def acc(rows, task=None):
        ks = [v[0] for k, v in rows.items() if task is None or k[0] == task]; return sum(ks) / max(1, len(ks)), len(ks)
    print("BASE", sys.argv[1], {t: acc(B, t) for t in ("t1", "tox", "inj")})
    for p in sys.argv[2:]:
        C = ok_rows(json.load(open(p))); res = {"file": p, "acc": {t: acc(C, t) for t in ("t1", "tox", "inj")}}
        for task in ("t1", "tox", "inj", None):
            keep = [k for k in B if B[k][0] and k in C and (task is None or k[0] == task)]
            n = len(keep); a = sum(C[k][0] for k in keep); res[f"ret_{task or 'ALL'}"] = (a / max(1, n), n, round(wilson_lo(a, n), 3))
            cells = collections.defaultdict(lambda: [0, 0])
            for k in keep: c = cells[B[k][1]]; c[0] += C[k][0]; c[1] += 1
            if task is None: res["worst_cells"] = sorted(((round(a / n, 3), n, c) for c, (a, n) in cells.items() if n >= 20))[:5]
        print(json.dumps(res, default=str)); out[p] = res
    json.dump(out, open(os.path.dirname(os.path.abspath(sys.argv[1])) + "/analysis_" + os.path.basename(sys.argv[1]).replace(".json", "") + ".json", "w"), default=str, indent=1)
main()
