"""(2) Shipped CLIs from a clean copy of the artifact: answer.py (handful of items) -> feedback.py (12 Recipe + 13 other annotations, writes fb.bin) -> answer.py -> feedback.py with empty ann -> answer.py.
Neighbours: 7 old-label probes + 40 dev (sel) items answered before / after / after-empty; Recipe items taught. Also a {text,label} annotation file."""
import sys, os, json, argparse, subprocess, shutil, time
ap = argparse.ArgumentParser(); ap.add_argument("--out"); ap.add_argument("--root", default="/workspace/fin"); ap.add_argument("--src", default="/r/art"); a = ap.parse_args()
ROOT = a.root; sys.path.insert(0, ROOT + "/v2/exam"); import loader as L
art = "/tmp/art"; shutil.rmtree(art, ignore_errors=True); shutil.copytree(a.src, art)
CODE = ROOT + "/v2/lanes/L12"
for f in ("l12_runtime.py", "answer.py", "feedback.py", "annot.py", "skill_task.json", "feedback_cal.jsonl"): shutil.copy(f"{CODE}/{f}", f"{art}/{f}")   # shipped code = repo copy
pool = L.load("SKILL7", "ann_pool"); sel = L.load("SKILL7", "sel")
neigh = [r for r in sel[::45]][:40]; T = json.load(open(f"{art}/skill_task.json"))["prompt"]
probes_new = ["how do i bake chocolate chip cookies", "give me a recipe for lasagna", "what is the best way to cook a turkey", "how long do i boil an egg"]
probes_old = ["wake me up at seven", "what is the weather in paris", "play some jazz", "add milk to my shopping list", "when does the next bus leave", "put the dentist on my calendar for friday", "read me the headlines"]
texts = probes_new + probes_old + [r["user"].split("Request: ", 1)[1] for r in neigh]
def sh(cmd, tag):
    t = time.time(); r = subprocess.run(cmd, capture_output=True, text=True, cwd="/tmp"); print("E2E", tag, "rc", r.returncode, round(time.time() - t), "s", r.stdout[-600:], r.stderr[-400:] if r.returncode else "", flush=True)
    return dict(rc=r.returncode, secs=round(time.time() - t), stdout=r.stdout, stderr=r.stderr[-1500:] if r.returncode else "")
ans = lambda tag: sh(["python3", f"{art}/answer.py", art] + texts, tag)
res = {}; res["before"] = ans("answer_before")
# annotation file: 25 rows, schema {query,label}: first 12 Recipe (text form), 13 existing-label pool items
rec = ["how do i bake chocolate chip cookies cookies" if False else t for t in ["how long do i boil an egg", "what is a good substitute for butter in cookies", "give me a recipe for vegetable soup", "how do i make pancakes from scratch", "how much flour for one loaf of bread", "tell me how to cook rice on the stove", "what temperature do I bake chicken at", "steps to prepare a tomato sauce", "how do i marinate steak overnight", "what can i use instead of eggs in a cake", "recipe for homemade guacamole", "how long to simmer a beef stew"]]
old = [r for r in pool][:13]
with open("/tmp/ann.jsonl", "w") as f:
    for t in rec: f.write(json.dumps({"text": t, "label": "Recipe"}) + "\n")
    for r in old: f.write(json.dumps({"query": r["user"], "label": r["answer"]}) + "\n")
res["feedback"] = sh(["python3", f"{art}/feedback.py", "--artifact", art, "--ann", "/tmp/ann.jsonl"], "feedback")
res["fb_bin_bytes"] = os.path.getsize(f"{art}/fb.bin") if os.path.exists(f"{art}/fb.bin") else None; res["fb_json_bytes"] = os.path.getsize(f"{art}/fb.json") if os.path.exists(f"{art}/fb.json") else None
res["after"] = ans("answer_after"); shutil.copy(f"{art}/fb.bin", a.out + "/fb_recipe.bin")
res["empty"] = sh(["python3", f"{art}/feedback.py", "--artifact", art, "--ann", "/dev/null"], "feedback_empty"); res["fb_exists_after_empty"] = os.path.exists(f"{art}/fb.bin")
res["after_empty"] = ans("answer_after_empty")
b, af, ae = [res[k]["stdout"].strip().split("\n") for k in ("before", "after", "after_empty")]
n1 = len(probes_new); n2 = n1 + len(probes_old)
res["analysis"] = dict(texts=texts, before=b, after=af, after_empty=ae, new_before=b[:n1], new_after=af[:n1], old_probes_changed=sum(x != y for x, y in zip(b[n1:n2], af[n1:n2])), neigh_changed=sum(x != y for x, y in zip(b[n2:], af[n2:])), n_neigh=len(b[n2:]),
    neigh_gold=[r["answer"] for r in neigh], neigh_before_correct=sum(x == r["answer"] for x, r in zip(b[n2:], neigh)) / len(neigh), neigh_after_correct=sum(x == r["answer"] for x, r in zip(af[n2:], neigh)) / len(neigh),
    restored_identical=(ae == b), recipe_taught=sum(x == "Recipe" for x in af[:n1]), recipe_leak_on_old=sum(x == "Recipe" for x in af[n1:]))
# inventory of shipped bytes with fb.bin of the recipe run
json.dump(res, open(a.out + "/cli.json", "w"), indent=1); print("E2E CLI", json.dumps(res["analysis"])); print("E2E DONE")
