"""usage: tf_analyze.py tf.json [more.json...]  -> teacher-forced retention table (conditioned on base ok) with Wilson CI."""
import json, sys, math
def wil(k, n, z=1.96):
    if n == 0: return (0, 0)
    p = k / n; d = 1 + z*z/n; c = p + z*z/(2*n); r = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n)); return ((c-r)/d, (c+r)/d)
R = {}
for f in sys.argv[1:]: R.update(json.load(open(f)))
base = R["base"]; ids = [i for i in base if base[i]["ok"]]
print(f"base ok {len(ids)}/{len(base)}")
print("cfg | retention all [CI] | S1 | S3 | nocall | single | multi | mean tokacc | mean nll")
for c, d in R.items():
    if c == "base": continue
    def r(f):
        s = [i for i in ids if f(base[i])]; k = sum(d[i]["ok"] for i in s); return f"{k/max(len(s),1):.3f}({k}/{len(s)})"
    k = sum(d[i]["ok"] for i in ids); lo, hi = wil(k, len(ids))
    print(f"{c} | {k/len(ids):.3f} [{lo:.2f},{hi:.2f}] | {r(lambda b:b['source']=='S1')} | {r(lambda b:b['source']=='S3')} | " + " | ".join(r(lambda b,k=k:b['kind']==k) for k in ("nocall","single","multi")) + f" | {sum(d[i]['tokacc'] for i in d)/len(d):.3f} | {sum(d[i]['nll'] for i in d)/len(d):.3f}")
