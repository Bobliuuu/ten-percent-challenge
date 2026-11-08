"""Feedback data for L5: new skill `Kitchen` (CLINC kitchen_dining intents) kept out of every prompt menu. DEV only (bucket [0,70)); never exam.
Writes fb_data/{fb_annot_25,fb_teach_unseen_200,items_dev,calib}.jsonl in the format of v2/fb_edit/data_news."""
import json, hashlib, re, random, collections, os
from datasets import load_dataset
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = HERE + "/fb_data"; os.makedirs(OUT, exist_ok=True)
H = lambda s: hashlib.sha256(s.encode()).hexdigest()
norm = lambda t: re.sub(r"[^a-z0-9 ]", "", " ".join(str(t).lower().split()))
bucket = lambda t: int(H("easy-split-v1|SKILL7|" + norm(t)), 16) % 100
KI = {"recipe", "ingredients_list", "cook_time", "nutrition_info", "calories", "ingredient_substitution", "food_last"}
LAB = ["Alarm", "Weather", "Music", "Calendar", "Lists", "News", "Transport"]
P = "You are the router of a voice assistant. Decide which skill should handle the user's request.\nAnswer with exactly one word from: " + ", ".join(LAB) + ".\n\nRequest: {X}"
ks = {}
for sp, ds in load_dataset("DeepPavlov/clinc_oos", name="plus").items():
    for i, x in enumerate(ds):
        if x["label_text"] in KI: ks.setdefault(norm(x["text"]), (f"k{sp}{i}", x["label_text"], x["text"]))
K = [v for v in ks.values() if bucket(v[2]) < 70]; K.sort(key=lambda v: H(v[0])); print("kitchen items bucket<70:", len(K), collections.Counter(v[1] for v in K))
# annotation: 25 items spread over intents (round robin); teach: next 200 distinct, disjoint
by = collections.defaultdict(list)
for v in K: by[v[1]].append(v)
ann = []; i = 0
while len(ann) < 25:
    for it in sorted(by):
        if len(ann) < 25 and by[it]: ann.append(by[it].pop(0))
rest = [v for it in sorted(by) for v in by[it]]; random.Random(0).shuffle(rest); teach = rest[:200]
row = lambda pre, v: dict(id=f"K|{pre}|{v[0]}", task="SKILL7", role=pre, src="clinc150", kind="Kitchen", user=P.replace("{X}", v[2]), answer="Kitchen")
w = lambda n, L: open(f"{OUT}/{n}.jsonl", "w").write("".join(json.dumps(r) + "\n" for r in L))
w("fb_annot_25", [row("ann", v) for v in ann]); w("fb_teach_unseen_200", [row("teach", v) for v in teach])
dev = [json.loads(l) for l in open(HERE + "/data_s7big/dev.jsonl")]; random.Random(1).shuffle(dev)
cnt = collections.Counter(); nb = []
for r in dev:
    c = (r["src"], r["kind"])
    if cnt[c] < 120: cnt[c] += 1; nb.append(r)
w("items_dev", nb); cal = [json.loads(l) for l in open(HERE + "/data_s7big/calib.jsonl")]; w("calib", cal)
print(len(ann), len(teach), len(nb), len(cal))
