"""END-TO-END test of the SHIPPED commands on a copy of the artifact dir (SKILL7 + new skill Recipe): feedback.py CLI -> patch.pt -> answer.py CLI real generation.
python e2e_skill.py ART WORKDIR EXTRA OUT.json     arms: real / shuffled / empty. Prints + writes: teach acc (unseen Recipe, source A and B), neighbours kept, tree sha, bytes."""
import sys, os, json, subprocess, shutil, hashlib, random, collections
art, work, extra, out = sys.argv[1:5]; HERE = os.path.dirname(os.path.abspath(__file__)); H = lambda s: int(hashlib.sha256(s.encode()).hexdigest()[:8], 16)
rd = lambda p: [json.loads(l) for l in open(p)]
raw = lambda r: r["user"].split("Request: ", 1)[1]
A = sorted(rd(extra + "/skill_new_a.jsonl"), key=lambda r: H("fbx-ent|" + r["id"])); B = sorted(rd(extra + "/skill_new_b.jsonl"), key=lambda r: H("fbx-bbc|" + r["id"]))
ann = random.Random(1000 * 25).sample(A[:60], 25); teachA = A[100:]; teachB = B[40:140]
items = [r for r in rd(HERE + "/../../task_search/EASY/items_dev.jsonl") if r["task"] == "SKILL7"]
pool = [r for r in rd(extra + "/skill_old.jsonl") if r["kind"]][:25]
os.makedirs(work, exist_ok=True)
def sha(d):
    h = hashlib.sha256()
    for f in sorted(os.listdir(d)):
        if os.path.isfile(d + "/" + f): h.update(f.encode()); h.update(open(d + "/" + f, "rb").read())
    return h.hexdigest()[:16]
def answer(d, rows):
    p = work + "/in.jsonl"; open(p, "w").write("".join(json.dumps({"text": raw(r)}) + "\n" for r in rows))
    o = subprocess.run([sys.executable, d + "/answer.py", "--artifact", d, "--task", d + "/skill_task.json", "--in", p], capture_output=True, text=True); assert o.returncode == 0, o.stderr[-800:]
    return o.stdout.rstrip("\n").split("\n")
res = {}
base = answer(art, items); ok = [r for r, o in zip(items, base) if o == r["kind"]]; res["base_acc"] = len(ok) / len(items)
neigh = ok[::2]
for arm in ("real", "shuffled", "empty"):
    d = work + "/art_" + arm
    if os.path.exists(d): shutil.rmtree(d)
    shutil.copytree(art, d, symlinks=True)
    rows = [dict(text=raw(r), label="Recipe") for r in ann] if arm == "real" else ([dict(text=raw(r), label="Recipe") for r in pool] if arm == "shuffled" else [])
    open(work + "/ann.jsonl", "w").write("".join(json.dumps(r) + "\n" for r in rows)); before = sha(art)
    o = subprocess.run([sys.executable, d + "/feedback.py", "--artifact", d, "--task", d + "/skill_task.json", "--ann", work + "/ann.jsonl"], capture_output=True, text=True); assert o.returncode == 0, o.stderr[-800:]
    ta = answer(d, teachA); tb = answer(d, teachB); nb = answer(d, neigh)
    res[arm] = dict(feedback_stdout=o.stdout.strip(), tree_sha_before=before, tree_sha_after=sha(d), teach_A=sum(x == "Recipe" for x in ta) / len(ta), teach_B=sum(x == "Recipe" for x in tb) / len(tb),
                    neigh_kept=sum(x == r["kind"] for x, r in zip(nb, neigh)) / len(neigh), n_neigh=len(neigh), patch_bytes=os.path.getsize(d + "/patch.pt") if os.path.exists(d + "/patch.pt") else 0,
                    sample=[(raw(r)[:60], x) for r, x in list(zip(teachA, ta))[:6]], off_menu_outputs=dict(collections.Counter(x for x in ta + tb if x not in json.load(open(d + "/skill_task.json"))["labels"]).most_common(4)))
    print(arm, json.dumps(res[arm])[:600], flush=True); json.dump(res, open(out, "w"), indent=1)
    if arm != "real": shutil.rmtree(d)
