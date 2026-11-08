"""Build the voice-assistant routing task (mimics a voice-assistant routing benchmark) from MASSIVE en-US (public).
7 old skills (weather/email/transport/lists/calendar/home(=iot)/news), 2 NEW skills recipe(=cooking) and music.
Splits by sha256(id): teach set (locked, hash written to data_route/LOCK.json BEFORE any model run). Dev/pool data only."""
import json, hashlib, os, tarfile, urllib.request, collections, random, sys
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data_route"); os.makedirs(OUT, exist_ok=True)
tgz = "/tmp/massive.tgz"
if not os.path.exists(tgz): urllib.request.urlretrieve("https://amazon-massive-nlu-dataset.s3.amazonaws.com/amazon-massive-dataset-1.1.tar.gz", tgz)
R = [json.loads(l) for l in tarfile.open(tgz).extractfile("1.1/data/en-US.jsonl")]
MAP = {"weather": "weather", "email": "email", "transport": "transport", "lists": "lists", "calendar": "calendar", "iot": "home", "news": "news", "cooking": "recipe", "music": "music"}
OLD = ["weather", "email", "transport", "lists", "calendar", "home", "news"]; NEWL = ["recipe", "music"]
H = lambda s: int(hashlib.sha256(s.encode()).hexdigest()[:8], 16) % 100
seen = set(); items = []
for r in R:
    if r["scenario"] not in MAP: continue
    t = r["utt"].strip(); k = t.lower()
    if k in seen: continue
    seen.add(k); items.append(dict(id="massive-" + r["id"] + "-" + r["partition"], text=t, gold=MAP[r["scenario"]], src=r["partition"], grp="massive-" + r["id"] + "-" + r["partition"], intent=r["intent"]))
by = collections.defaultdict(list)
for it in items: by[it["gold"]].append(it)
W = lambda n, rows: open(f"{OUT}/{n}.jsonl", "w").write("".join(json.dumps(r) + "\n" for r in rows))
nw = {"nw_ann": [], "nw_dev": [], "nw_teach": [], "nw_ood": []}
for y in NEWL:
    for it in by[y]:
        h = H("split" + it["id"])
        nw["nw_ann" if h < 25 else "nw_dev" if h < 40 else "nw_ood" if h < 55 else "nw_teach"].append(it)
for k, v in nw.items(): W(k, v)
rng = random.Random(0); old = [it for y in OLD for it in by[y]]
test = [it for it in old if it["src"] in ("test", "dev")]; train = [it for it in old if it["src"] == "train"]
cells = collections.defaultdict(list)
for it in test: cells[(it["src"], it["gold"])].append(it)
ne = []
for k in sorted(cells): v = sorted(cells[k], key=lambda x: x["id"]); rng.shuffle(v); ne += v[:150]
W("neigh_eval", ne); ids = {x["id"] for x in ne}
rest = [x for x in test if x["id"] not in ids]; rng.shuffle(rest); W("neigh_alpha", rest[:200])
train = sorted(train, key=lambda x: x["id"]); rng.shuffle(train)
W("other_pool", train[:750]); W("calib_text", train[750:1750])
lock = {n: hashlib.sha256(open(f"{OUT}/{n}", "rb").read()).hexdigest() for n in sorted(os.listdir(OUT)) if n.endswith(".jsonl")}
json.dump(lock, open(f"{OUT}/LOCK.json", "w"), indent=1)
print({k: len(v) for k, v in nw.items()}, "neigh_eval", len(ne), "calib", 1000, collections.Counter((x["src"], x["gold"]) for x in ne))
print(lock["nw_teach.jsonl"])
