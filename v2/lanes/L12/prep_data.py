"""L12 data prep: SKILL7 voice-skill routing (same prompt/hash scheme as v2/task_search/EASY/build_easy.py).
calib = hash bucket [0,40) (capped), SEL = [40,55) (configuration choice), REP = [55,70) (reporting), [70,100) untouched (locked).
Feedback task FBHOME: new skill 'Home' (MASSIVE iot scenario) not in the prompt label list: annot 25 (bucket [0,20)), teach (bucket [40,70)).
python prep_data.py -> data/*.jsonl"""
import json, hashlib, re, random, collections, os
from datasets import load_dataset
H = lambda s: hashlib.sha256(s.encode()).hexdigest()
norm = lambda t: re.sub(r"[^a-z0-9 ]", "", " ".join(str(t).lower().split()))
bucket = lambda task, t: int(H(f"easy-split-v1|{task}|" + norm(t)), 16) % 100
def trunc(t, n): t = " ".join(str(t).split()); return t if len(t) <= n else t[:n] + " ..."
LABELS = ["Alarm", "Weather", "Music", "Calendar", "Lists", "News", "Transport"]
P = "You are the router of a voice assistant. Decide which skill should handle the user's request.\nAnswer with exactly one word from: " + ", ".join(LABELS) + ".\n\nRequest: {X}"
ms = load_dataset("mteb/amazon_massive_scenario", name="en", split="train")
SK = {"alarm": "Alarm", "weather": "Weather", "music": "Music", "calendar": "Calendar", "lists": "Lists", "news": "News", "transport": "Transport"}
r1 = [(f"m{x['id']}", SK[x["label_text"]], x["text"]) for x in ms if x["label_text"] in SK]
iot = [(f"m{x['id']}", "Home", x["text"]) for x in ms if x["label_text"] == "iot"]
clinc = load_dataset("DeepPavlov/clinc_oos", name="plus", split="train")
def mp(d): return {k: v for v, ks in d.items() for k in ks}
CL = mp({"Alarm": ["alarm", "timer"], "Weather": ["weather"], "Music": ["play_music", "next_song", "what_song", "update_playlist"], "Calendar": ["calendar", "calendar_update", "meeting_schedule", "schedule_meeting"],
         "Lists": ["todo_list", "todo_list_update", "shopping_list", "shopping_list_update"], "Transport": ["directions", "traffic", "uber", "distance"]})
r2 = [(f"c{i}", CL[x["label_text"]], x["text"]) for i, x in enumerate(clinc) if x["label_text"] in CL]
def build(task, srcs):
    out = collections.defaultdict(list)
    for src, rows, tl in srcs:
        seen = set()
        for rid, lab, text in rows:
            k = norm(text)
            if k in seen or not k: continue
            seen.add(k); b = bucket(task, text)
            out[b].append(dict(id=f"{task}|{src}|{rid}", src=src, kind=lab, user=P.replace("{X}", trunc(text, tl * 3)), answer=lab, b=b))
    return out
sk = build("SKILL7", [("massive", r1, 300), ("clinc150", r2, 300)])
cal, sel, rep = [], [], []
for b, rows in sk.items():
    for r in rows: (cal if b < 40 else sel if b < 55 else rep if 55 <= b < 70 else []).append(r)
rnd = random.Random(0); rnd.shuffle(cal)
cnt = collections.Counter(); cal2 = []
for r in cal:
    k = (r["src"], r["kind"])
    if cnt[k] < 100: cnt[k] += 1; cal2.append(r)
os.makedirs("data", exist_ok=True)
w = lambda n, rows: open(f"data/{n}.jsonl", "w").write("".join(json.dumps(r) + "\n" for r in rows))
w("cal", cal2); w("sel", sorted(sel, key=lambda r: r["id"])); w("rep", sorted(rep, key=lambda r: r["id"]))
fh = build("FBHOME", [("massive_iot", iot, 300)])
ann = sorted([r for b, rows in fh.items() if b < 20 for r in rows], key=lambda r: H(r["id"]))[:25]
teach = sorted([r for b, rows in fh.items() if 40 <= b < 70 for r in rows], key=lambda r: H(r["id"]))[:200]
w("fb_annot", ann); w("fb_teach", teach)
print("cal", len(cal2), "sel", len(sel), "rep", len(rep), "fb", len(ann), len(teach))
for n, rows in (("cal", cal2), ("sel", sel), ("rep", rep)): print(n, sorted(collections.Counter(r["src"] + "|" + r["kind"] for r in rows).items()))
