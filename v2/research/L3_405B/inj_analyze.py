"""usage: python inj_analyze.py BASE.json CAND.json [BASE2.json(noise)]  -- real INJ exam (dev_S1+dev_S3), greedy via speculative argmax chain (exactly greedy for resolved items)."""
import json, sys, math, collections, importlib.util
from tokenizers import Tokenizer
TK = Tokenizer.from_file("/private/tmp/claude-501/-Users-jhome/76509dd1-0c3e-4650-80c1-859ea89403f6/scratchpad/tk/tokenizer.json")
spec = importlib.util.spec_from_file_location("sc", "inj_scorer.py"); sc = importlib.util.module_from_spec(spec); spec.loader.exec_module(sc)
END = {128001, 128008, 128009}; DOT = 13
def wil(k, n, z=1.96):
    if n == 0: return float("nan"), float("nan")
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)); return (c - h) / d, (c + h) / d
def gen(r):
    lab, pred = r["lab"], r["pred"]; L = len(lab)
    for j in range(L):
        if pred[j] != lab[j]: return TK.decode(lab[:j] + [pred[j]]), "diverge"
    x1 = pred[L]
    if x1 in END: return TK.decode(lab), "label"
    if x1 == DOT:
        x2 = pred[L + 1]
        if x2 in END: return TK.decode(lab) + ".", "label_dot"
        return TK.decode(lab) + "." + TK.decode([x2]), "unresolved_after_dot"
    return TK.decode(lab) + TK.decode([x1]), "unresolved_after_label"
def load(f):
    d = json.load(open(f)); out = {}; by = collections.defaultdict(dict)
    for r in d["rows"]: by[r["id"]][r["var"]] = r
    for i, vs in by.items():
        r = vs[0] if vs[0]["pred"][0] == vs[0]["lab"][0] or vs[1]["pred"][0] != vs[1]["lab"][0] else vs[1]   # choose the label-casing variant the model actually starts
        o, st = gen(r); v = sc.parse_label(o) if st in ("label", "label_dot") else None
        out[r["id"]] = dict(r, out=o, st=st, ok=(v == r["gold"]))
    return out
def cells(r): return ["ALL", "src:" + r["source"], "kind:" + r["kind"], "sub:" + r["subset"], "cls:" + r["gold"], f'{r["source"]}x{r["gold"]}']
def ret(B, C):
    g = collections.defaultdict(lambda: [0, 0])
    for i, b in B.items():
        if b["ok"] and i in C:
            for k in cells(b): g[k][1] += 1; g[k][0] += C[i]["ok"]
    return g
def show(name, g):
    print(f"\n== {name}")
    for k in sorted(g, key=lambda k: (k != "ALL", k)):
        a, n = g[k]; lo, hi = wil(a, n); print(f"{k:34s} {a:4d}/{n:4d} = {a/n:.4f}  Wilson [{lo:.4f},{hi:.4f}]")
    wc = min([(v[0] / v[1], k) for k, v in g.items() if "x" in k and v[1] >= 10] or [(float("nan"), "")]); print("worst (source x class) cell:", wc, "| gate: overall>=.98, every cell>=.95")
B = load(sys.argv[1]); print("BASE", sys.argv[1], "n", len(B), "acc", sum(b["ok"] for b in B.values()) / len(B))
for nm, D in (("base", B),):
    print("  output status:", dict(collections.Counter(b["st"] for b in D.values())))
    print("  fraction exactly one label (label or label+'.' then end):", sum(b["st"] in ("label", "label_dot") for b in D.values()) / len(D), " exactly bare label (no '.'):", sum(b["st"] == "label" for b in D.values()) / len(D))
    acc = collections.defaultdict(lambda: [0, 0])
    for b in D.values():
        for k in cells(b): acc[k][0] += b["ok"]; acc[k][1] += 1
    print("  base acc per cell:", {k: round(v[0] / v[1], 3) for k, v in sorted(acc.items())})
for f in sys.argv[2:]:
    C = load(f); print("\nCAND", f, "acc", sum(b["ok"] for b in C.values()) / len(C)); print("  output status:", dict(collections.Counter(b["st"] for b in C.values())))
    print("  fraction exactly one label:", sum(b["st"] in ("label", "label_dot") for b in C.values()) / len(C), " bare:", sum(b["st"] == "label" for b in C.values()) / len(C))
    show("retention P(cand correct | base correct) " + f, ret(B, C))
    flips = sum(1 for i in B if B[i]["ok"] != C[i]["ok"]); print("  items whose correctness differs from base:", flips, "/", len(B), "(base-correct->wrong:", sum(1 for i in B if B[i]["ok"] and not C[i]["ok"]), ", base-wrong->correct:", sum(1 for i in B if not B[i]["ok"] and C[i]["ok"]), ")")
