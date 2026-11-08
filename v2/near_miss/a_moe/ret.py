"""ret.py base.json cand.json... : SKILL7 accuracy + retention P(cand ok|base ok), Wilson lo, per (src,kind) cells (n>=20 min cell). exact one-word label, case-insens, punctuation stripped."""
import json, sys, re, math, collections
LAB = ["Alarm","Weather","Music","Calendar","Lists","News","Transport"]
def ok(o, g): return re.sub(r"[^A-Za-z]", "", o.split("\n")[0].strip()).lower() == g.lower()
def wl(k, n, z=1.96):
    if n == 0: return 0.0
    p = k/n; d = 1+z*z/n; c = p+z*z/(2*n); h = z*math.sqrt(p*(1-p)/n+z*z/(4*n*n)); return (c-h)/d
def rows(p): return {r["id"]: (ok(r["out"], r["gold"]), (r["src"], r["gold"])) for r in json.load(open(p))["rows"]}
def main():
    B = rows(sys.argv[1]); print("BASE acc", sum(v[0] for v in B.values())/len(B), len(B))
    for p in sys.argv[2:]:
        C = rows(p); keep = [k for k in B if B[k][0] and k in C]; a = sum(C[k][0] for k in keep)
        cells = collections.defaultdict(lambda: [0, 0])
        for k in keep: c = cells[B[k][1]]; c[0] += C[k][0]; c[1] += 1
        mc = min([(a_/n, c) for c, (a_, n) in cells.items() if n >= 20] or [(float('nan'), None)])
        print(json.dumps(dict(f=p.split("/")[-1], acc=round(sum(v[0] for v in C.values())/len(C), 4), ret=round(a/len(keep), 4), n=len(keep), wlo=round(wl(a, len(keep)), 3), mincell=(round(mc[0], 3), mc[1]))))
main()
