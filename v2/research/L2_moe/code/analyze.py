"""usage: analyze.py DIR  -> table of acc and item retention vs base.json (overall/per kind/per source) with Wilson lower bounds."""
import json, sys, glob, os, math, collections
def wil(k, n, z=1.96):
    if n == 0: return (0, 0)
    p = k / n; d = 1 + z*z/n; c = p + z*z/(2*n); r = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))
    return ((c - r)/d, (c + r)/d)
D = sys.argv[1]; base = json.load(open(f"{D}/base.json"))
bc = {r["id"]: r for r in base["rows"]}
print(f"base acc {base['acc']:.3f}  n={len(bc)}")
print("cfg | acc | ret all [wilsonlo] | S1 | S3 | nocall | single | multi | parsefail")
for f in sorted(glob.glob(f"{D}/*.json")):
    if f.endswith("base.json"): continue
    d = json.load(open(f)); rows = d["rows"]
    def ret(flt):
        bs = [r for r in rows if bc[r["id"]]["correct"] and flt(r)]
        k = sum(r["correct"] for r in bs); n = len(bs)
        return f"{k/max(n,1):.3f}({k}/{n})", wil(k, n)[0]
    a, alo = ret(lambda r: True)
    print(f"{d['cfg']} | {d['acc']:.3f} | {a} [{alo:.3f}] | {ret(lambda r:r['source']=='S1')[0]} | {ret(lambda r:r['source']=='S3')[0]} | " + " | ".join(ret(lambda r,k=k:r['kind']==k)[0] for k in ("nocall","single","multi")) + f" | {sum(r['parse_fail'] for r in rows)/len(rows):.3f}")
