"""SKILL7 extended dev/calib for the expert-quantiser sweep. Same prompt/label map/hash buckets as task_search/EASY/build_easy.py but larger caps. calib=bucket[0,40), dev=bucket[40,70); [70,100) untouched."""
import json, hashlib, collections, re
from datasets import load_dataset
H = lambda s: hashlib.sha256(s.encode()).hexdigest(); norm = lambda t: re.sub(r"[^a-z0-9 ]", "", " ".join(str(t).lower().split()))
bucket = lambda t: int(H("easy-split-v1|SKILL7|" + norm(t)), 16) % 100
def trunc(t, n): t = " ".join(str(t).split()); return t if len(t) <= n else t[:n] + " ..."
def mp(d): return {k: v for v, ks in d.items() for k in ks}
ms = load_dataset("mteb/amazon_massive_scenario", name="en", split="train")
SK = {"alarm": "Alarm", "weather": "Weather", "music": "Music", "calendar": "Calendar", "lists": "Lists", "news": "News", "transport": "Transport"}
r1 = [(f"m{x['id']}", SK[x["label_text"]], x["text"]) for x in ms if x["label_text"] in SK]
clinc = load_dataset("DeepPavlov/clinc_oos", name="plus", split="train")
CL = mp({"Alarm": ["alarm", "timer"], "Weather": ["weather"], "Music": ["play_music", "next_song", "what_song", "update_playlist"], "Calendar": ["calendar", "calendar_update", "meeting_schedule", "schedule_meeting"],
         "Lists": ["todo_list", "todo_list_update", "shopping_list", "shopping_list_update"], "Transport": ["directions", "traffic", "uber", "distance"]})
r2 = [(f"c{i}", CL[x["label_text"]], x["text"]) for i, x in enumerate(clinc) if x["label_text"] in CL]
labels = ["Alarm", "Weather", "Music", "Calendar", "Lists", "News", "Transport"]
P = "You are the router of a voice assistant. Decide which skill should handle the user's request.\nAnswer with exactly one word from: " + ", ".join(labels) + ".\n\nRequest: {X}"
dev, cal = [], []
for src, rows, tl in (("massive", r1, 300), ("clinc150", r2, 300)):
    seen = set(); D = collections.defaultdict(list); C = collections.defaultdict(list)
    for rid, lab, text in rows:
        k = norm(text)
        if k in seen or not k: continue
        seen.add(k); b = bucket(text)
        if 40 <= b < 70: D[lab].append((H("d" + src + rid), rid, text))
        elif b < 40: C[lab].append((H("c" + src + rid), rid, text))
    for lab in D:
        for _, rid, t in sorted(D[lab])[:400]: dev.append(dict(id=f"SKILL7|{src}|{rid}", src=src, kind=lab, user=P.replace("{X}", trunc(t, tl * 3)), answer=lab))
        for _, rid, t in sorted(C[lab])[:200]: cal.append(dict(id=f"SKILL7|{src}|cal|{rid}", src=src, kind=lab, user=P.replace("{X}", trunc(t, tl * 3)), answer=lab))
import random; random.Random(0).shuffle(cal); random.Random(1).shuffle(dev)
open("data/skill7_sel.jsonl", "w").write("".join(json.dumps(i) + "\n" for i in dev)); open("data/skill7_cal.jsonl", "w").write("".join(json.dumps(i) + "\n" for i in cal))
print(len(dev), len(cal), collections.Counter((d["src"], d["kind"]) for d in dev))
