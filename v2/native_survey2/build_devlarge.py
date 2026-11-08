"""Build dev-large eval set (no locked/audit files). Reads data from the main checkout (tournament data dirs are untracked). Deterministic seed 0.
usage: python3 build_devlarge.py OUT.jsonl"""
import json, random, sys, os, collections, hashlib
R = random.Random(0); TV = "/Users/jhome/ten-percent/v2/tournament"; EX = "/Users/jhome/tp-c27/v2/exp/splits"; NL = "/Users/jhome/tp-c27/v2/native_lowbit"
TOXP = open(NL + "/tox_prompt.txt").read(); INJP = open(NL + "/inj_prompt.txt").read()
def rd(p): return [json.loads(l) for l in open(p)]
out = []; seen = set()
def take(rows, keyf, n, mk):
    cells = collections.defaultdict(list)
    for r in rows: cells[keyf(r)].append(r)
    for k, v in sorted(cells.items()):
        R.shuffle(v)
        for r in v[:n]:
            x = mk(r)
            if x["id"] in seen: continue
            seen.add(x["id"]); out.append(x)
# TOX: dev + pool, 700 per (source, class)
tox = []
for f in ("s1_dev", "s3_dev", "s1_pool", "s3_pool"): tox += rd(f"{TV}/TOX/data/{f}.jsonl")
take(tox, lambda r: (r["source"], r["gold"]), 700, lambda r: dict(id="tox|" + r["id"], task="tox", source=r["source"], kind=r["gold"], gold=r["gold"], user=TOXP.replace("{t}", r["text"]), maxnew=16))
inj = []
for f in ("dev_S1", "dev_S3", "pool_S1", "calib_S1"): inj += rd(f"{TV}/INJ/data/{f}.jsonl")
take(inj, lambda r: (r["source"], r["gold"]), 700, lambda r: dict(id="inj|" + r["id"], task="inj", source=r["source"], kind=r["gold"], gold=r["gold"], user=INJP.replace("{t}", r["text"]), maxnew=16))
t1 = []
for f in ("s1_dev", "s3_dev", "s1_pool", "s3_pool"): t1 += rd(f"{TV}/T1/data/{f}.jsonl")
take(t1, lambda r: (r["source"], r["kind"]), 400, lambda r: dict(id="t1|" + r["id"], task="t1", source=r["source"], kind=r["kind"], gold=r["gold"], user=r["query"], tools=r["tools"], maxnew=300))
news = rd(EX + "/sel.jsonl") + rd(EX + "/rep.jsonl")
for r in news:
    if r["id"] in seen: continue
    seen.add(r["id"]); out.append(dict(id=r["id"], task="news", source=r["src"], kind=r["kind"], gold=r["answer"], user=r["user"], maxnew=12))
with open(sys.argv[1], "w") as f:
    for x in out: f.write(json.dumps(x) + "\n")
print(collections.Counter((x["task"], x["source"], x["kind"]) for x in out).most_common(60)); print(len(out), hashlib.sha256(open(sys.argv[1], "rb").read()).hexdigest())
