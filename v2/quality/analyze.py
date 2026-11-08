"""retention table: python analyze.py base.json cand1.json ...   (files from Modal volume tenpct-quality:/res, copied to results/)"""
import json, sys, collections, math, numpy as np
def wil(k, n, z=1.96):
    if n == 0: return 0.0
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)); return (c - h) / d
def ret(B, C, cells):
    bm = {r["id"]: r for r in B}; cm = {r["id"]: r for r in C}; g = collections.defaultdict(lambda: [0, 0])
    for i, b in bm.items():
        if b["correct"]:
            for key in ["ALL"] + ["/".join(cells(b))] : g[key][1] += 1; g[key][0] += cm[i]["correct"]
    a = g["ALL"]; cs = {k: v[0] / v[1] for k, v in g.items() if k != "ALL" and v[1] >= 20}
    return a[0] / a[1], wil(*a), min(cs.values()), min(cs, key=cs.get)
def load(f): return json.load(open(f))
if __name__ == "__main__":
    base = load(sys.argv[1]); print("| run | ratio(fixed/ent) | bpw | INJ ret (wilson lo) worst cell | TOOLSEL ret (lo) worst cell | MMLU/ARC/HS/TQA ret | ppl | KL |"); 
    for f in sys.argv[2:]:
        c = load(f); a = ret(base["real_inj"], c["real_inj"], lambda b: [tuple(b["cell"])[1]] if False else [str(b["cell"][0]) + "_" + str(b["cell"][1])]) if "real_inj" in c else None
        t = ret(base["real_toolsel"], c["real_toolsel"], lambda b: [str(b["cell"][0]) + "_" + str(b["cell"][1])]) if "real_toolsel" in c else None
        gs = c.get("general_summary", {}); gr = "/".join(f"{gs[k]['retention']*100:.0f}" for k in gs)
        fm = lambda x: f"{x[0]*100:.1f} ({x[1]*100:.1f}) {x[2]*100:.0f}@{x[3]}" if x else "-"
        print(f"| {c['name']} | {c.get('ratio_fixed',0)*100:.2f}/{c.get('ratio_ent',0)*100:.2f} | {c.get('body_bpw_fixed',0):.3f} | {fm(a)} | {fm(t)} | {gr} | {c['general']['ppl']:.2f} | {c['general']['kl']:.3f} |")
