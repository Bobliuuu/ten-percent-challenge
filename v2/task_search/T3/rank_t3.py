"""python rank_t3.py : reads summary_t3.json (written by analyze_t3.py) + PLAN*.json, prints ranked markdown (easy-first)."""
import json, collections, glob
S = json.load(open("summary_t3.json")); PL = {}
for f in glob.glob("PLAN*.json"): PL.update(json.load(open(f)))
rows = collections.defaultdict(dict)
for k, v in S.items():
    tagset, cfg, task = k.split("|"); model = tagset.split("/")[0].replace("-Instruct", "").replace("Qwen2.5-", "Q2.5-")
    rows[task][model] = v
def f(x): return "-" if x is None else f"{x:.3f}"
out = ["| task | classes | sources | base acc 14B / 32B | worst-class base 14B | margin>=6 (14B) | retention 14B @9.85% [lo] (worst kind) | retention 32B @9.85% [lo] (worst kind) |", "|---|---|---|---|---|---|---|---|"]
def key(t):
    r = rows[t]; m = r.get("Q2.5-14B") or r.get("Q2.5-32B") or {}
    return -(m.get("lo", 0))
for t in sorted(rows, key=key):
    r = rows[t]; a = r.get("Q2.5-14B"); b = r.get("Q2.5-32B"); pl = PL.get(t, {})
    nc = len(pl.get("labels", [])) or "?"; ns = len(pl.get("src", {})) or "?"
    out.append(f"| {t} | {nc} | {ns} | {f(a and a['base'])} / {f(b and b['base'])} | {f(a and a['worstclass'])} | {f(a and a['m6'])} | " + (f"{a['ret']:.3f} [{a['lo']:.3f}] ({a['wkind']:.2f})" if a else "-") + " | " + (f"{b['ret']:.3f} [{b['lo']:.3f}] ({b['wkind']:.2f})" if b else "-") + " |")
print("\n".join(out))
