"""offline scorer: python score.py RESDIR -> table of retention vs bytes for every voice_<name>.json (re-scores outs against outs_{DS}_base_SEL.json)"""
import json, sys, glob, os, math, collections
d = sys.argv[1]; DS = sys.argv[2] if len(sys.argv) > 2 else "skill7"; rows = [json.loads(l) for l in open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "moe_quant", "data", f"{DS}_sel.jsonl"))]
def wil(k, n, z=1.96):
    if n == 0: return 0.0
    p = k / n; dd = 1 + z * z / n; c = p + z * z / (2 * n); h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)); return (c - h) / dd
b = json.load(open(f"{d}/outs_{DS}_base_SEL.json")); assert b["ids"] == [r["id"] for r in rows]
bc = [o == r["answer"] for o, r in zip(b["outs"], rows)]; nb = sum(bc); out = []
for f in sorted(glob.glob(f"{d}/{DS}_*.json")):
    j = json.load(open(f)); n = j["name"]
    if n == "base": continue
    o = json.load(open(f"{d}/outs_{DS}_{n}_SEL.json"))["outs"]; ok = [x == r["answer"] for x, r in zip(o, rows)]
    k = sum(1 for x, y in zip(ok, bc) if x and y); cells = collections.defaultdict(lambda: [0, 0])
    for x, y, r in zip(ok, bc, rows):
        if y: c = cells[r["kind"]]; c[0] += 1; c[1] += int(x)
    mc = min(v[1] / v[0] for v in cells.values()); mcl = min(wil(v[1], v[0]) for v in cells.values())
    qi = j.get("qinfo") or {}
    out.append((n, j["cfg"].get("m"), j["cfg"].get("keep"), j["bytes"]["ratio"], qi.get("expert_bpw"), sum(ok) / len(ok), k / nb, wil(k, nb), mc, qi.get("proxy_relerr_gu"), qi.get("proxy_relerr_dn")))
print(f"base acc {nb/len(rows):.4f} n_bc {nb}")
print("| name | method | keep/layer | bytes % fp16 | expert bpw | acc | retention | Wilson lo | min-kind ret | relerr gu | relerr dn |\n|---|---|---|---|---|---|---|---|---|---|---|")
f = lambda x, p=4: "" if x is None else f"{x:.{p}f}"
for r in sorted(out, key=lambda r: -r[6]): print(f"| {r[0]} | {r[1]} | {r[2]} | {100*r[3]:.2f} | {f(r[4],3)} | {r[5]:.4f} | {r[6]:.4f} | {r[7]:.4f} | {r[8]:.3f} | {f(r[9])} | {f(r[10])} |")
