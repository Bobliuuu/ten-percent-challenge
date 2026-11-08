"""Locked-style SKILL7 exam for L9b. Pool = MASSIVE en (train+validation+test) and CLINC150 plus (train+validation+test), sha256 bucket [70,100) only (calib [0,40) / dev [40,70) never overlap).
Same prompt / label map as task_search/EASY/build_easy.py. Output exam.jsonl + EXAM_MANIFEST.json (sha256 of the file, counts). Never used for building or choosing."""
import json, hashlib, collections, re, random
from datasets import load_dataset, concatenate_datasets
H = lambda s: hashlib.sha256(s.encode()).hexdigest(); norm = lambda t: re.sub(r"[^a-z0-9 ]", "", " ".join(str(t).lower().split()))
bucket = lambda t: int(H("easy-split-v1|SKILL7|" + norm(t)), 16) % 100
def trunc(t, n): t = " ".join(str(t).split()); return t if len(t) <= n else t[:n] + " ..."
def mp(d): return {k: v for v, ks in d.items() for k in ks}
SK = {"alarm": "Alarm", "weather": "Weather", "music": "Music", "calendar": "Calendar", "lists": "Lists", "news": "News", "transport": "Transport"}
CL = mp({"Alarm": ["alarm", "timer"], "Weather": ["weather"], "Music": ["play_music", "next_song", "what_song", "update_playlist"], "Calendar": ["calendar", "calendar_update", "meeting_schedule", "schedule_meeting"],
         "Lists": ["todo_list", "todo_list_update", "shopping_list", "shopping_list_update"], "Transport": ["directions", "traffic", "uber", "distance"]})
labels = ["Alarm", "Weather", "Music", "Calendar", "Lists", "News", "Transport"]
P = "You are the router of a voice assistant. Decide which skill should handle the user's request.\nAnswer with exactly one word from: " + ", ".join(labels) + ".\n\nRequest: {X}"
pool = collections.defaultdict(list)
for sp in ("train", "validation", "test"):
    for x in load_dataset("mteb/amazon_massive_scenario", name="en", split=sp):
        if x["label_text"] in SK: pool[("massive", SK[x["label_text"]])].append((f"m{sp[0]}{x['id']}", x["text"]))
    for i, x in enumerate(load_dataset("DeepPavlov/clinc_oos", name="plus", split=sp)):
        if x["label_text"] in CL: pool[("clinc150", CL[x["label_text"]])].append((f"c{sp[0]}{i}", x["text"]))
seen = set(); items = []; rng = random.Random(20261108)
cap = {"massive": 330, "clinc150": 130}
for (src, lab), rows in sorted(pool.items()):
    cand = []
    for rid, t in rows:
        k = norm(t)
        if not k or k in seen or bucket(t) < 70: continue
        seen.add(k); cand.append((H("x" + src + rid), rid, t))
    cand.sort()
    print(src, lab, len(cand))
    for _, rid, t in cand[:cap[src]]: items.append(dict(id=f"SKILL7X|{src}|{rid}", src=src, kind=lab, user=P.replace("{X}", trunc(t, 900 if src == "massive" else 900)), answer=lab))
rng.shuffle(items)
open("exam.jsonl", "w").write("".join(json.dumps(i) + "\n" for i in items))
sha = hashlib.sha256(open("exam.jsonl", "rb").read()).hexdigest()
cnt = collections.Counter(i["kind"] for i in items); srcs = collections.Counter(i["src"] for i in items)
json.dump(dict(file="exam.jsonl", sha256=sha, n=len(items), per_kind=cnt, per_src=srcs, bucket="[70,100)", sealed="2026-11-08, built before any base/compressed run on it"), open("EXAM_MANIFEST.json", "w"), indent=1)
print(len(items), cnt, srcs, sha)
