"""mk_data.py OUTROOT: builds the dataset dirs fb_l2x.py reads (ds_recipe, ds_jukebox, ds_briefing, ds_reminder). Run from v2/lanes/L2.
ds_reminder (second feedback label, NOT in the exam, NOT in the menu): CLINC150 intents reminder + reminder_update (not mapped by v2/exam/pools.py, so disjoint from the exam by construction);
ann from train, teach from validation+test, texts deduped against ann."""
import json, os, sys, re, random, shutil
R = sys.argv[1]; D = "data"; X = "data/exam"; T = json.load(open("skill_task.json"))
rd = lambda f: [json.loads(l) for l in open(f)]
def wr(d, n, rows): os.makedirs(f"{R}/{d}", exist_ok=True); open(f"{R}/{d}/{n}.jsonl", "w").write("".join(json.dumps(r) + "\n" for r in rows))
old = rd(f"{D}/skill7_probe.jsonl")
for r in old: r.setdefault("src", "dev")
wr("ds_recipe", "ann", rd(f"{X}/ann_Recipe.jsonl")); wr("ds_recipe", "teach", rd(f"{X}/teach_Recipe.jsonl")); wr("ds_recipe", "neigh", rd(f"{X}/neigh.jsonl")); wr("ds_recipe", "cal", rd(f"{X}/cal.jsonl")); wr("ds_recipe", "old", old)
json.dump({"new": "Recipe"}, open(f"{R}/ds_recipe/DATASET.json", "w"))
for nm in ("Jukebox", "Briefing"):
    wr(f"ds_{nm.lower()}", "ann", rd(f"{D}/ann_{nm}.jsonl")); wr(f"ds_{nm.lower()}", "teach", rd(f"{D}/teach_{nm}.jsonl")); wr(f"ds_{nm.lower()}", "neigh", rd(f"{D}/neigh.jsonl")); wr(f"ds_{nm.lower()}", "cal", rd(f"{D}/cal.jsonl"))
    json.dump({"new": nm, "shuf_labels": ["Alarm", "Weather", "Calendar", "Lists", "Transport"]}, open(f"{R}/ds_{nm.lower()}/DATASET.json", "w"))
from datasets import load_dataset
ds = load_dataset("DeepPavlov/clinc_oos", name="plus"); rows = {}
norm = lambda t: re.sub(r"[^a-z0-9 ]", "", " ".join(t.lower().split()))
for sp, d in ds.items():
    for i, x in enumerate(d):
        if x["label_text"] in ("reminder", "reminder_update"): rows.setdefault(sp, []).append(dict(id=f"clinc150|rem|{sp}{i}", src="clinc150", kind="Reminder", answer="Reminder", user=T["prompt"].replace("{t}", x["text"]), text=x["text"]))
ann = rows["train"]; seen = {norm(r["text"]) for r in ann}; teach = []
for r in rows.get("validation", []) + rows["test"]:
    if norm(r["text"]) not in seen: teach.append(r); seen.add(norm(r["text"]))
wr("ds_reminder", "ann", ann); wr("ds_reminder", "teach", teach); json.dump({"new": "Reminder"}, open(f"{R}/ds_reminder/DATASET.json", "w"))
print({sp: len(v) for sp, v in rows.items()}, "ann", len(ann), "teach", len(teach))
