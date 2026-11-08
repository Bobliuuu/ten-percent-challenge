"""SKILL7 retention (base-correct kept) per draw: overall + Wilson lo, per (src,kind) cell, per kind, SEL/REP split (md5(id)%2).
usage: python analyze_s7.py BASE_ct.json CAND_ct.json [CAND2 ...]   -> prints; with several cands also mean/min over draws."""
import json, sys, math, hashlib, collections, os, statistics as st
def wil(k, n, z=1.96):
    if n == 0: return float("nan")
    p = k / n; d = 1 + z * z / n; return (p + z * z / (2 * n) - z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / d
split = lambda i: "SEL" if int(hashlib.md5(i.encode()).hexdigest(), 16) % 2 == 0 else "REP"
def load(p): return {r["id"]: r for r in json.load(open(p))["rows"]}
def ret(B, C, part="ALL"):
    cells = collections.defaultdict(lambda: [0, 0])
    for i, b in B.items():
        if not b["correct"] or i not in C: continue
        if part != "ALL" and split(i) != part: continue
        for key in ("ALL", b["src"] + "_" + b["kind"], "kind:" + b["kind"], "src:" + b["src"]):
            cells[key][1] += 1; cells[key][0] += int(C[i]["correct"])
    return cells
if __name__ == "__main__":
    B = load(sys.argv[1]); print("base acc", sum(r["correct"] for r in B.values()) / len(B), "n", len(B))
    allc = []
    for p in sys.argv[2:]:
        C = load(p); c = ret(B, C); allc.append(c); a, n = c["ALL"]
        cc = {k: v for k, v in c.items() if "_" in k and not k.startswith(("kind:", "src:")) and v[1] >= 30}
        w = min(cc.items(), key=lambda kv: kv[1][0] / kv[1][1])
        kk = {k: v[0] / v[1] for k, v in c.items() if k.startswith("kind:")}
        print(f"{os.path.basename(p)}: ret {a/n:.4f} lo {wil(a,n):.4f} n={n} | worst src_kind {w[0]} {w[1][0]/w[1][1]:.3f} (n={w[1][1]}) | cells<95%: {sum(1 for v in cc.values() if v[0]/v[1]<.95)}/{len(cc)} | kinds " + " ".join(f"{k[5:]}={v:.3f}" for k, v in sorted(kk.items())))
    if len(allc) > 1:
        r = [c["ALL"][0] / c["ALL"][1] for c in allc]; print(f"draws {len(r)}: mean {st.mean(r):.4f} min {min(r):.4f} max {max(r):.4f}")
        keys = [k for k in allc[0] if "_" in k and not k.startswith(("kind:", "src:"))]
        for k in sorted(keys): 
            v = [c[k][0] / c[k][1] for c in allc]; print(f"  {k:22s} n={allc[0][k][1]:4d} mean {st.mean(v):.3f} min {min(v):.3f}")
