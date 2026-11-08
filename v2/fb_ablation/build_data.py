"""L11 data: SKILL7-derived, 5 old skills (Alarm Weather Calendar Lists Transport) + 2 NEW labels (Music->Jukebox, News->Briefing). Dev/calib buckets only ([0,70)); [70,100) never read.
bucket = sha256('easy-split-v1|SKILL7|norm(text)')%100 (same as task_search/EASY). calib [0,40) = key covariance + shuffle pool; ann [40,55); teach [55,70); neighbours: old classes in [40,70) (dev).
Outputs data/*.jsonl"""
import json, hashlib, re, random, collections
from datasets import load_dataset
H = lambda s: hashlib.sha256(s.encode()).hexdigest()
norm = lambda t: re.sub(r"[^a-z0-9 ]", "", " ".join(str(t).lower().split()))
bucket = lambda t: int(H("easy-split-v1|SKILL7|" + norm(t)), 16) % 100
def trunc(t, n): t = " ".join(str(t).split()); return t if len(t) <= n else t[:n] + " ..."
ms = load_dataset("mteb/amazon_massive_scenario", name="en", split="train")
SK = {"alarm": "Alarm", "weather": "Weather", "music": "Music", "calendar": "Calendar", "lists": "Lists", "news": "News", "transport": "Transport"}
r1 = [("massive", f"m{x['id']}", SK[x["label_text"]], x["text"]) for x in ms if x["label_text"] in SK]
clinc = load_dataset("DeepPavlov/clinc_oos", name="plus", split="train")
def mp(d): return {k: v for v, ks in d.items() for k in ks}
CL = mp({"Alarm": ["alarm", "timer"], "Weather": ["weather"], "Music": ["play_music", "next_song", "what_song", "update_playlist"], "Calendar": ["calendar", "calendar_update", "meeting_schedule", "schedule_meeting"],
         "Lists": ["todo_list", "todo_list_update", "shopping_list", "shopping_list_update"], "Transport": ["directions", "traffic", "uber", "distance"]})
r2 = [("clinc150", f"c{i}", CL[x["label_text"]], x["text"]) for i, x in enumerate(clinc) if x["label_text"] in CL]
OLD = ["Alarm", "Weather", "Calendar", "Lists", "Transport"]; NEWMAP = {"Music": "Jukebox", "News": "Briefing"}
seen = set(); rows = []
for src, rid, lab, text in r1 + r2:
    k = norm(text)
    if not k or k in seen: continue
    seen.add(k); b = bucket(text)
    if b >= 70: continue
    rows.append(dict(id=f"{src}|{rid}", src=src, gold=lab, text=trunc(text, 300), b=b))
cap = lambda L, n, seed: random.Random(seed).sample(L, min(n, len(L)))
out = collections.defaultdict(list)
for r in rows:
    b = r["b"]; lab = r["gold"]
    if lab in NEWMAP:
        r["new"] = NEWMAP[lab]
        out["ann_" + lab if 40 <= b < 55 else "teach_" + lab if 55 <= b < 70 else "calnew_" + lab].append(r)
    else:
        out["calib_old" if b < 40 else "neigh_old"].append(r)
# neighbours: cap 60 per (src, gold) cell for speed; calib old: 400 total spread
neigh = []
by = collections.defaultdict(list)
for r in out["neigh_old"]: by[(r["src"], r["gold"])].append(r)
for k in sorted(by): neigh += cap(by[k], 80, 1)
calib = cap(out["calib_old"], 600, 2)
def dump(n, L): open(f"data/{n}.jsonl", "w").write("".join(json.dumps(r) + "\n" for r in L)); print(n, len(L), collections.Counter((r["src"], r["gold"]) for r in L).most_common(12))
dump("neigh", neigh); dump("calib_old", calib)
for k in ["ann_Music", "teach_Music", "ann_News", "teach_News", "calnew_Music", "calnew_News"]: dump(k, sorted(out[k], key=lambda r: r["id"]))
