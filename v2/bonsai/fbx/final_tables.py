"""python final_tables.py RESULT.json BASEPREDS.json [skill|news]  -> markdown tables (controls, per-cell neighbour retention, proxy arms, policy)"""
import sys, json, hashlib, numpy as np, collections
R = json.load(open(sys.argv[1])); BASE = json.load(open(sys.argv[2])); task = sys.argv[3] if len(sys.argv) > 3 else "skill"
HERE = __import__("os").path.dirname(__import__("os").path.abspath(__file__)); H = lambda s: int(hashlib.sha256(s.encode()).hexdigest()[:8], 16)
it = [json.loads(l) for l in open(HERE + ("/../../task_search/EASY/items_dev.jsonl" if task == "skill" else "/../../fb_edit/data_news/items_dev.jsonl"))]
it = [r for r in it if r.get("task") == "SKILL7"] if task == "skill" else it
NB = [r for r in it if BASE[r["id"]] == r["kind"]]; REP = [r for r in NB if H("fbx-nb|" + r["id"]) % 2 == 1]
def ci(v, B=3000):
    v = np.array(v, float); rr = np.random.default_rng(0)
    return [round(float(np.percentile([rr.choice(v, len(v)).mean() for _ in range(B)], q)), 3) for q in (2.5, 97.5)] if len(v) > 1 else None
f = lambda L: f"{np.mean(L):.3f} {ci(L)}" if len(L) else "-"
runs = R["runs"]; by = collections.defaultdict(list)
for r in runs: by[(r["arm"], r["n"])].append(r)
mean = lambda rs, k: [float(np.mean(r["res"][k]["ok"])) for r in rs if k in r["res"] and r["res"][k]["ok"]]
print("| arm | n_ann | draws | teach A (unseen, same source as annotations) | teach B (other source) | neighbours kept | worst draw | MB |\n|---|---|---|---|---|---|---|---|")
for (arm, n), rs in by.items():
    nk = mean(rs, "neigh_rep"); print(f"| {arm} | {n} | {len(rs)} | {f(mean(rs,'teach_rep'))} | {f(mean(rs,'teach_bbc'))} | {f(nk)} | {min(nk) if nk else '-'} | {rs[0]['bytes']/1e6:.1f} |")
cells = collections.defaultdict(list)
for r in by.get(("real", 25), []):
    ok = r["res"]["neigh_rep"]["ok"]; assert len(ok) == len(REP), (len(ok), len(REP))
    for row, o in zip(REP, ok): cells[(row["src"], row["kind"])].append(o)
print("\nper-cell neighbour retention (real, n=25, all draws pooled):"); print({f"{k[0]}|{k[1]}": [round(float(np.mean(v)), 3), len(v)] for k, v in sorted(cells.items())})
per_src = collections.defaultdict(list)
for (s, k), v in cells.items(): per_src[s] += v
print("per source:", {s: round(float(np.mean(v)), 4) for s, v in per_src.items()})
re_ = {r["d"]: r for r in by.get(("real", 25), [])}; sh = {r["d"]: r for r in by.get(("shuffled", 25), [])}
gap = [np.mean(re_[d]["res"]["teach_rep"]["ok"]) - np.mean(sh[d]["res"]["teach_rep"]["ok"]) for d in re_ if d in sh]; print("real - shuffled (teach A) per draw:", f(gap))
