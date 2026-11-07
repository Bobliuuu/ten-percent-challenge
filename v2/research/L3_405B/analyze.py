"""usage: python analyze.py BASE.json CAND.json [CAND2.json]   (rows need m_all). correct := every LABEL token (all answer tokens except the final <|eot_id|>) is the argmax.
Llama-3.1 prefers '.' over <|eot_id|> after a bare label, so EOT is excluded (decode would still parse as the label). Also reports strict (incl. EOT)."""
import json, sys, math, collections
def wil(k, n, z=1.96):
    if n == 0: return 0.0
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)); return (c - h) / d
def load(f, strict=False):
    d = json.load(open(f)); return {r["id"]: dict(r, ok=(all(m > 0 for m in r["m_all"]) if strict else all(m > 0 for m in r["m_all"][:-1]))) for r in d["rows"]}
strict = "--strict" in sys.argv; a = [x for x in sys.argv[1:] if not x.startswith("--")]
B = load(a[0], strict); print("base acc per task:", {t: round(sum(r["ok"] for r in B.values() if r["task"] == t) / sum(1 for r in B.values() if r["task"] == t), 3) for t in sorted({r["task"] for r in B.values()})})
for f in a[1:]:
    C = load(f, strict); print("\n==", f, "criterion:", "strict(incl EOT)" if strict else "label tokens")
    for t in sorted({r["task"] for r in B.values()}):
        g = collections.defaultdict(lambda: [0, 0])
        for i, b in B.items():
            if b["task"] == t and b["ok"]:
                for key in ("ALL", "role:" + b["role"], "src:" + b["src"], "kind:" + b["kind"]): g[key][1] += 1; g[key][0] += C[i]["ok"]
        cells = [v for k, v in g.items() if k != "ALL"]; allv = g["ALL"]
        if not allv[1]: continue
        ret = allv[0] / allv[1]; wc = [(v[0] / v[1], k) for k, v in g.items() if k != "ALL" and v[1] >= 10]; worst = min(wc)
        ok = ret >= .98 and wil(*allv) >= .98 and worst[0] >= .95
        print(f"{t:8s} base-correct {allv[1]:3d}  ret {ret:.3f} (lo {wil(*allv):.3f}; {allv[0]}/{allv[1]}) worst cell {worst[0]:.3f} [{worst[1]}] PASS={ok} | " + "; ".join(f"{k} {v[0]/v[1]:.2f}({v[1]})" for k, v in sorted(g.items()) if k != "ALL" and k.startswith(("role", "kind")) and len(g) < 40))
