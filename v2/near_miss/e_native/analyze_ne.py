"""analyze_ne.py RESULTS_DIR TAG : retention of every candidate vs the model's own base (P(cand ok | base ok)), per task + min cell (n>=20), Wilson lower bounds. Uses repo scorers."""
import sys, json, os, glob, math, collections, importlib.util
V2 = os.path.expanduser("~/ten-percent/v2")
def load(p, n):
    s = importlib.util.spec_from_file_location(n, p); m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
T1 = load(V2 + "/tournament/T1/scorer.py", "t1s"); TOX = load(V2 + "/tournament/TOX/scorer.py", "toxs"); INJ = load(V2 + "/tournament/INJ/scorer.py", "injs")
items = {json.loads(l)["id"]: json.loads(l) for l in open(V2 + "/tournament/T1/probe_items.jsonl")}
def wl(k, n, z=1.96):
    if n == 0: return 0.0
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)); return (c - h) / d
def lab(out, gold): return out.replace("<|eot_id|>", "").replace("<|im_end|>", "").strip().strip(".").strip().lower() == gold.lower()
def rows(R):
    r = {}
    for x in R["t1"]:
        try: ok = bool(T1.score_item(x["out"], items[x["id"]])["correct"])
        except Exception: ok = False
        r[("t1", x["id"])] = (ok, ("t1", x["source"], x["kind"]))
    for x in R["cls"]:
        ok = TOX.correct(x["out"], x["gold"]) if x["task"] == "tox" else INJ.correct(x["out"], x["gold"]); r[(x["task"], x["id"])] = (ok, (x["task"], x["source"], x["gold"]))
    for t in ("skill7", "bank8"):
        for x in R.get(t, []): r[(t, x["id"])] = (lab(x["out"], x["gold"]), (t, x["source"], x["kind"]))
    return r
def main(d, tag):
    B = rows(json.load(open(f"{d}/{tag}__base.json"))); out = []
    for t in ("t1", "tox", "inj", "skill7", "bank8"):
        ks = [v[0] for k, v in B.items() if k[0] == t]
        if ks: print(f"BASE {tag} {t}: acc {sum(ks)/len(ks):.3f} n={len(ks)}")
    for f in sorted(glob.glob(f"{d}/{tag}__*.json")):
        c = os.path.basename(f)[len(tag) + 2:-5]
        if c in ("base", "sizes", "cands"): continue
        C = rows(json.load(open(f))); res = {"cand": c}
        for t in ("t1", "tox", "inj", "skill7", "bank8", None):
            keep = [k for k in B if B[k][0] and k in C and (t is None or k[0] == t)]; n = len(keep); a = sum(C[k][0] for k in keep)
            res["ret_" + (t or "ALL")] = (round(a / max(1, n), 3), n, round(wl(a, n), 3))
            if t is None or t in ("skill7", "bank8"):
                cells = collections.defaultdict(lambda: [0, 0])
                for k in keep: cc = cells[B[k][1]]; cc[0] += C[k][0]; cc[1] += 1
                w = sorted((round(a_ / n_, 3), n_, k_) for k_, (a_, n_) in cells.items() if n_ >= 20)
                res["minc_" + (t or "ALL")] = w[:1]
        out.append(res); print(json.dumps(res))
    json.dump(out, open(f"{d}/analysis_{tag}.json", "w"), indent=1)
main(sys.argv[1], sys.argv[2])
