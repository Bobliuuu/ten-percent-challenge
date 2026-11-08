"""L9 locked-style exam builder (SKILL7). Items = hash bucket [70,100) of the SAME bucket function as v2/task_search/EASY/build_easy.py / lanes/L12/prep_data.py, i.e. disjoint
from calibration (b<40), SEL (40-55) and REP (55-70). Pools: MASSIVE (scenario, train+validation+test) and CLINC150-plus (train+validation+test).
python build_exam.py -> data/exam_skill7.jsonl + EXAM_SEAL.json (sha256 of file, counts). Written ONCE; never evaluated until the freeze. Not an X15/LOCK file."""
import json, hashlib, re, collections
from datasets import load_dataset, concatenate_datasets
H = lambda s: hashlib.sha256(s.encode()).hexdigest(); norm = lambda t: re.sub(r"[^a-z0-9 ]", "", " ".join(str(t).lower().split()))
bucket = lambda t: int(H("easy-split-v1|SKILL7|" + norm(t)), 16) % 100
def trunc(t, n): t = " ".join(str(t).split()); return t if len(t) <= n else t[:n] + " ..."
LABELS = ["Alarm", "Weather", "Music", "Calendar", "Lists", "News", "Transport"]
P = "You are the router of a voice assistant. Decide which skill should handle the user's request.\nAnswer with exactly one word from: " + ", ".join(LABELS) + ".\n\nRequest: {X}"
SK = {"alarm": "Alarm", "weather": "Weather", "music": "Music", "calendar": "Calendar", "lists": "Lists", "news": "News", "transport": "Transport"}
CL = {k: v for v, ks in {"Alarm": ["alarm", "timer"], "Weather": ["weather"], "Music": ["play_music", "next_song", "what_song", "update_playlist"], "Calendar": ["calendar", "calendar_update", "meeting_schedule", "schedule_meeting"],
      "Lists": ["todo_list", "todo_list_update", "shopping_list", "shopping_list_update"], "Transport": ["directions", "traffic", "uber", "distance"]}.items() for k in ks}
rows = []
for sp in ("train", "validation", "test"):
    for x in load_dataset("mteb/amazon_massive_scenario", name="en", split=sp):
        if x["label_text"] in SK: rows.append(("massive", f"m{sp[0]}{x['id']}", SK[x["label_text"]], x["text"]))
    for i, x in enumerate(load_dataset("DeepPavlov/clinc_oos", name="plus", split=sp)):
        if x["label_text"] in CL: rows.append(("clinc150", f"c{sp[0]}{i}", CL[x["label_text"]], x["text"]))
seen = set(); out = []
for src, rid, lab, text in sorted(rows, key=lambda r: H(r[1])):
    k = norm(text)
    if not k or k in seen: continue
    seen.add(k)
    if bucket(text) >= 70: out.append(dict(id=f"EXAM|{src}|{rid}", src=src, kind=lab, user=P.replace("{X}", trunc(text, 900)), answer=lab))
open("data/exam_skill7.jsonl", "w").write("".join(json.dumps(r) + "\n" for r in out))
c = collections.Counter((r["src"], r["kind"]) for r in out); k = collections.Counter(r["kind"] for r in out); s = collections.Counter(r["src"] for r in out)
seal = dict(sha256=hashlib.sha256(open("data/exam_skill7.jsonl", "rb").read()).hexdigest(), n=len(out), per_kind=dict(k), per_source=dict(s), per_cell={f"{a}|{b}": v for (a, b), v in sorted(c.items())},
            checks=dict(n_ge_2000=len(out) >= 2000, min_kind_ge_300=min(k.values()) >= 300, other_source_ge_500=min(s.values()) >= 500))
json.dump(seal, open("EXAM_SEAL.json", "w"), indent=1); print(json.dumps(seal, indent=1))
