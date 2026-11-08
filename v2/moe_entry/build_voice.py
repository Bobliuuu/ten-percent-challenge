"""Build the voice-assistant SKILL ROUTING task (MASSIVE en-US + CLINC150 plus), splits CAL/SEL/REP/LOCK + feedback sets. Deterministic.
Run: python3 -I build_voice.py   (raw files from v2/tournament/TOX/routing/data/raw)"""
import json, re, glob, hashlib, collections, os, pandas as pd
R = os.path.dirname(os.path.abspath(__file__)); RAW = os.path.abspath(R + "/../tournament/TOX/routing/data/raw"); OUT = R + "/data"; os.makedirs(OUT, exist_ok=True)
SK = {"clock": "alarms, timers, current time and date", "calendar": "calendar events, meetings, reminders", "lists": "to-do lists and shopping lists", "email": "reading, sending and managing email and contacts",
      "smart_home": "smart home devices, lights, plugs, thermostats, volume and device settings", "music": "playing music, radio, podcasts, audiobooks and games", "weather": "weather forecasts",
      "transport": "taxis, trains, flights, directions, traffic and distance", "food": "food delivery, restaurants, recipes, cooking and nutrition", "qa": "general knowledge questions, news, definitions and facts"}
MA = {"alarm": "clock", "datetime": "clock", "calendar": "calendar", "lists": "lists", "email": "email", "iot": "smart_home", "music": "music", "play": "music", "weather": "weather", "transport": "transport",
      "takeaway": "food", "cooking": "food", "qa": "qa", "news": "qa"}
CL = {"clock": "alarm timer time date timezone", "calendar": "calendar calendar_update reminder reminder_update meeting_schedule schedule_meeting", "lists": "todo_list todo_list_update shopping_list shopping_list_update",
      "smart_home": "smart_home sync_device change_volume reset_settings whisper_mode change_speed", "music": "play_music what_song next_song update_playlist", "weather": "weather",
      "transport": "uber directions traffic distance flight_status book_flight car_rental", "food": "recipe restaurant_suggestion restaurant_reviews meal_suggestion nutrition_info ingredients_list cook_time calories",
      "qa": "definition fun_fact next_holiday spelling"}
CLMAP = {i: q for q, s in CL.items() for i in s.split()}; NEWK = {"calculator", "measurement_conversion"}
PROMPT = ("You are the skill router of a voice assistant. Route the user's request to exactly one skill.\nSkills:\n" + "\n".join(f"- {k}: {v}" for k, v in SK.items()) +
          "\nAnswer with only the skill name.\n\nRequest: {X}")
norm = lambda t: re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", t.lower())).strip()
bucket = lambda t: int(hashlib.sha256(("v1|" + " ".join(norm(t).split()[:6])).encode()).hexdigest()[:8], 16) % 100
rows = []; seen = set()
for l in open(glob.glob(RAW + "/m/*/data/en-US.jsonl")[0]):
    r = json.loads(l)
    if r["scenario"] in MA and norm(r["utt"]) not in seen: seen.add(norm(r["utt"])); rows.append(dict(id="V|massive|" + r["id"], src="massive", kind=MA[r["scenario"]], text=r["utt"], orig=r["intent"]))
names = [n for _, n in re.findall(r"^\s+'?(\d+)'?: '?([\w]+)'?$", open(RAW + "/clinc__clinc_oos/README.md").read(), re.M)][:151]
for sp in ("train", "validation", "test"):
    d = pd.read_parquet(f"{RAW}/clinc__clinc_oos/plus/{sp}-00000-of-00001.parquet")
    for i, (t, y) in enumerate(zip(d.text, d.intent)):
        n = names[y]
        if (n in CLMAP or n in NEWK) and norm(t) not in seen:
            seen.add(norm(t)); rows.append(dict(id=f"V|clinc|{sp}{i}", src="clinc", kind=CLMAP.get(n, "NEW:" + n), text=t, orig=n))
mk = lambda r: dict(id=r["id"], src=r["src"], kind=r["kind"], orig=r["orig"], user=PROMPT.replace("{X}", r["text"]), answer=r["kind"], text=r["text"])
S = collections.defaultdict(list); srt = lambda xs: sorted(xs, key=lambda r: hashlib.sha256(("o|" + r["id"]).encode()).hexdigest())
for r in rows:
    b = bucket(r["text"])
    if r["kind"].startswith("NEW:"): S["fb_pool"].append(r); continue
    if r["src"] == "massive": S["cal" if b < 25 else "sel" if b < 40 else "rep" if b < 55 else "lockpool"].append(r)
    else: S["rep" if b < 15 else "lockpool"].append(r)
# LOCK: 330 per skill, clinc up to 100 per skill, rest massive
lock = []
for k in SK:
    cl = srt([r for r in S["lockpool"] if r["src"] == "clinc" and r["kind"] == k])[:100]; ma = srt([r for r in S["lockpool"] if r["src"] == "massive" and r["kind"] == k])[:330 - len(cl)]
    lock += cl + ma
S["lock"] = lock
fb = srt(S["fb_pool"]); calc = [r for r in fb if r["orig"] == "calculator"]; conv = [r for r in fb if r["orig"] == "measurement_conversion"]
out = {"cal": srt(S["cal"])[:3000], "sel": srt(S["sel"]), "rep": srt(S["rep"]), "lock": lock}
out["fb_annot24"] = calc[:24]; out["fb_teach_unseen"] = calc[24:124] + conv[:100]
for n, v in out.items():
    with open(f"{OUT}/voice_{n}.jsonl", "w") as f:
        for r in v: f.write(json.dumps(mk(r)) + "\n")
man = {n: dict(n=len(v), sha256=hashlib.sha256(open(f"{OUT}/voice_{n}.jsonl", "rb").read()).hexdigest(), cells=dict(collections.Counter(r["src"] + "|" + r["kind"] for r in v))) for n, v in out.items()}
json.dump(dict(skills=SK, prompt=PROMPT, files=man), open(OUT + "/VOICE_SPLITS.json", "w"), indent=1)
for n, m in man.items(): print(n, m["n"], m["sha256"][:12])
print(collections.Counter(r["kind"] for r in lock), collections.Counter(r["src"] for r in lock))
