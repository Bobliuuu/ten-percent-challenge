"""SKILL7 big dev/calib/exam builder. Same split rule as task_search/EASY/build_easy.py:
bucket = sha256('easy-split-v1|SKILL7|'+normalised text) % 100. dev/calib: [0,70) only; exam: [70,100) only.
MASSIVE en-US (mteb parquet mirror, all splits) and CLINC150 plus (DeepPavlov parquet, all splits).
Exam is built here but never scored; run once, at the end.
python build_s7big.py  (run from anywhere)"""
import json, hashlib, collections, re, os, random
from datasets import load_dataset
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "data_s7big"); os.makedirs(OUT, exist_ok=True)
OLD = os.path.join(HERE, "..", "..", "task_search", "EASY", "calib.jsonl")
H = lambda s: hashlib.sha256(s.encode()).hexdigest()
norm = lambda t: re.sub(r"[^a-z0-9 ]", "", " ".join(str(t).lower().split()))
bucket = lambda t: int(H("easy-split-v1|SKILL7|" + norm(t)), 16) % 100
def trunc(t, n): t = " ".join(str(t).split()); return t if len(t) <= n else t[:n] + " ..."
LABELS = ["Alarm", "Weather", "Music", "Calendar", "Lists", "News", "Transport"]
P = "You are the router of a voice assistant. Decide which skill should handle the user's request.\nAnswer with exactly one word from: " + ", ".join(LABELS) + ".\n\nRequest: {X}"
SK = {"alarm": "Alarm", "weather": "Weather", "music": "Music", "calendar": "Calendar", "lists": "Lists", "news": "News", "transport": "Transport"}
CL = {k: v for v, ks in {"Alarm": ["alarm", "timer"], "Weather": ["weather"], "Music": ["play_music", "next_song", "what_song", "update_playlist"], "Calendar": ["calendar", "calendar_update", "meeting_schedule", "schedule_meeting"],
      "Lists": ["todo_list", "todo_list_update", "shopping_list", "shopping_list_update"], "Transport": ["directions", "traffic", "uber", "distance"]}.items() for k in ks}
r1, r2 = [], []
for sp, ds in load_dataset("mteb/amazon_massive_scenario", name="en").items():
    r1 += [(f"m{sp}{x['id']}", SK[x["label_text"]], x["text"]) for x in ds if x["label_text"] in SK]
for sp, ds in load_dataset("DeepPavlov/clinc_oos", name="plus").items():
    r2 += [(f"c{sp}{i}", CL[x["label_text"]], x["text"]) for i, x in enumerate(ds) if x["label_text"] in CL]
old = set()
for l in open(OLD):
    j = json.loads(l)
    if j["task"] == "SKILL7": old.add(norm(j["user"].split("Request: ", 1)[1].removesuffix(" ...")))
oldn = set(old)
dev, cal, exam = [], [], []; excluded_old = 0; seen = set(); CAL_PER_CELL = 30
def row(role, src, rid, lab, text, pre): return dict(id=f"SKILL7|{src}|{pre}{rid}", task="SKILL7", role=role, src=src, kind=lab, user=P.replace("{X}", trunc(text, 900)), answer=lab)
for src, rows in (("massive", r1), ("clinc150", r2)):
    cells = collections.defaultdict(list)
    for rid, lab, text in rows:
        k = norm(text)
        if not k or k in seen: continue
        seen.add(k); b = bucket(text)
        if b < 70 and any(k == o or k.startswith(o) for o in ()): pass
        if b < 70 and k in oldn: excluded_old += 1; continue
        cells[lab].append((H(src + rid), b, rid, text))
    for lab, L in cells.items():
        L.sort(); lo = [x for x in L if x[1] < 70]; hi = [x for x in L if x[1] >= 70]
        for _, _, rid, t in lo[:CAL_PER_CELL]: cal.append(row("calib", src, rid, lab, t, "cal|"))
        for _, _, rid, t in lo[CAL_PER_CELL:]: dev.append(row("dev", src, rid, lab, t, ""))
        for _, _, rid, t in hi: exam.append(row("exam", src, rid, lab, t, "exam|"))
rnd = random.Random(0)
for L in (dev, cal, exam): rnd.shuffle(L)
tx = lambda L: {norm(r["user"].split("Request: ", 1)[1].removesuffix(" ...")) for r in L}
assert not (tx(dev) & tx(cal)) and not (tx(exam) & (tx(dev) | tx(cal) | oldn))
w = lambda n, L: open(os.path.join(OUT, n), "w").write("".join(json.dumps(r) + "\n" for r in L))
w("dev.jsonl", dev); w("calib.jsonl", cal); w("exam.jsonl", exam)
open(os.path.join(OUT, "exam.sha256"), "w").write(hashlib.sha256(open(os.path.join(OUT, "exam.jsonl"), "rb").read()).hexdigest() + "  exam.jsonl\n")
cnt = lambda L: collections.Counter((r["src"], r["kind"]) for r in L)
stats = {n: {f"{s}|{k}": v for (s, k), v in sorted(cnt(L).items())} for n, L in (("dev", dev), ("calib", cal), ("exam", exam))}
stats["totals"] = {n: len(L) for n, L in (("dev", dev), ("calib", cal), ("exam", exam))}
stats["old_calib_texts_excluded_from_pool"] = excluded_old
json.dump(stats, open(os.path.join(OUT, "stats.json"), "w"), indent=1); print(json.dumps(stats, indent=1))
