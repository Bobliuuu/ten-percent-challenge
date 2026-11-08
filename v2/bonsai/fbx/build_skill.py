"""DEV pools for the SKILL7 (voice-skill routing) feedback experiments. New skill `Recipe` = MASSIVE scenario cooking (source A) / CLINC150 cooking intents (source B). Buckets as build_easy.py (calib [0,40), dev [40,70), [70,100) never read).
  python build_skill.py --out DIR   -> skill_new_a.jsonl skill_new_b.jsonl skill_old.jsonl skill_calib.jsonl"""
import json, hashlib, re, argparse, os, collections
from datasets import load_dataset
H = lambda s: hashlib.sha256(s.encode()).hexdigest(); norm = lambda t: re.sub(r"[^a-z0-9 ]", "", " ".join(str(t).lower().split()))
bucket = lambda t: int(H("easy-split-v1|SKILL7|" + norm(t)), 16) % 100
ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
HERE = os.path.dirname(os.path.abspath(__file__)); EASY = HERE + "/../../task_search/EASY/"
P = json.load(open(EASY + "PLAN.json"))["SKILL7"]["prompt"]
used = {json.loads(l)["id"] for f in ("items_dev.jsonl", "calib.jsonl") for l in open(EASY + f) if json.loads(l)["task"] == "SKILL7"}
item = lambda src, rid, kind, text: dict(id=f"SKILL7|{src}|{rid}", src=src, kind=kind, user=P.replace("{X}", " ".join(text.split())[:900]))
ms = load_dataset("mteb/amazon_massive_scenario", name="en", split="train"); clinc = load_dataset("DeepPavlov/clinc_oos", name="plus", split="train")
SK = {"alarm": "Alarm", "weather": "Weather", "music": "Music", "calendar": "Calendar", "lists": "Lists", "news": "News", "transport": "Transport", "cooking": "Recipe"}
CK = {"recipe", "ingredients_list", "ingredient_substitution", "cook_time", "nutrition_info", "calories", "meal_suggestion"}
CL = {"alarm": "Alarm", "timer": "Alarm", "weather": "Weather", "play_music": "Music", "next_song": "Music", "what_song": "Music", "update_playlist": "Music", "calendar": "Calendar", "calendar_update": "Calendar", "meeting_schedule": "Calendar", "schedule_meeting": "Calendar",
      "todo_list": "Lists", "todo_list_update": "Lists", "shopping_list": "Lists", "shopping_list_update": "Lists", "directions": "Transport", "traffic": "Transport", "uber": "Transport", "distance": "Transport"}
seen = set(); A, B, O = [], [], []
for x in ms:
    k = norm(x["text"]); lab = SK.get(x["label_text"])
    if not lab or k in seen or bucket(x["text"]) >= 70: continue
    seen.add(k); it = item("massive", f"m{x['id']}", lab, x["text"])
    (A if lab == "Recipe" else O).append(it)
for i, x in enumerate(clinc):
    k = norm(x["text"])
    if k in seen or bucket(x["text"]) >= 70: continue
    seen.add(k)
    if x["label_text"] in CK: B.append(item("clinc150", f"c{i}", "Recipe", x["text"]))
    elif x["label_text"] in CL: O.append(item("clinc150", f"c{i}", CL[x["label_text"]], x["text"]))
O = [r for r in O if r["id"] not in used and bucket(r["user"].split("Request: ", 1)[1]) < 40 or r["id"] not in used]   # old pool: anything not in dev/calib, buckets < 70
O = [r for r in O if r["id"] not in used]
srt = lambda L, t: sorted(L, key=lambda r: H(t + r["id"]))
O = srt(O, "old")[:1400]; calib = [r for r in map(json.loads, open(EASY + "calib.jsonl")) if r["task"] == "SKILL7"]
cq = [dict(query=r["user"]) for r in calib]
w = lambda n, L: (open(a.out + "/" + n + ".jsonl", "w").write("".join(json.dumps(r) + "\n" for r in L)), print(n, len(L), collections.Counter(r.get("kind") for r in L)))
w("skill_new_a", srt(A, "a")); w("skill_new_b", srt(B, "b")); w("skill_old", O); w("skill_calib", cq)
