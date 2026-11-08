"""shipped-CLI protocol on the locked SKILL7 sets: feedback.py (<=25 annotations from ann_pool) -> patch.pt -> answer.py on locked teach + neighbours. arms real/shuffled/wrong/empty.
python e2e_locked.py ART WORKDIR TPROOT OUT.json"""
import sys, os, json, subprocess, shutil, hashlib, random, collections
art, work, root, out = sys.argv[1:5]; sys.path.insert(0, root + "/v2/exam"); import loader as LD
LD.verify(); teach = LD.load("SKILL7", "teach"); pool = LD.load("SKILL7", "ann_pool"); exam = {r["id"]: r for r in LD.load("SKILL7", "exam")}; sel = LD.load("SKILL7", "sel")
OLD = LD.labels("SKILL7"); raw = lambda r: r["user"].split("Request: ", 1)[1]; os.makedirs(work, exist_ok=True)
def sha(d):
    h = hashlib.sha256()
    for f in sorted(os.listdir(d)):
        if os.path.isfile(d + "/" + f): h.update(f.encode()); h.update(open(d + "/" + f, "rb").read())
    return h.hexdigest()[:16]
def answer(d, rows):
    p = work + "/in.jsonl"; open(p, "w").write("".join(json.dumps({"text": raw(r)}) + "\n" for r in rows))
    o = subprocess.run([sys.executable, d + "/answer.py", "--artifact", d, "--task", d + "/skill_task.json", "--in", p], capture_output=True, text=True); assert o.returncode == 0, o.stderr[-600:]
    return o.stdout.rstrip("\n").split("\n")
items = list(exam.values()); base = answer(art, items); okb = {r["id"]: o == r["kind"] for r, o in zip(items, base)}
neigh = [exam[i] for i in LD.load("SKILL7", "neighbours") if okb[i]]; rr = random.Random(77); ann = rr.sample(pool, 25)
sel_ok = [r for r in sel[:300] if r["kind"] in OLD][:25]; wrong = [rr.choice([l for l in OLD]) for _ in ann]
res = dict(candidate_exam_acc=sum(okb.values()) / len(okb))
for arm in ("real", "shuffled", "wrong", "empty"):
    d = work + "/art_" + arm
    if os.path.exists(d): shutil.rmtree(d)
    shutil.copytree(art, d, symlinks=True)
    rows = {"real": [(raw(r), "Recipe") for r in ann], "shuffled": [(raw(r), "Recipe") for r in sel_ok], "wrong": [(raw(r), w) for r, w in zip(ann, wrong)], "empty": []}[arm]
    open(work + "/ann.jsonl", "w").write("".join(json.dumps(dict(text=t, label=l)) + "\n" for t, l in rows)); before = sha(art)
    o = subprocess.run([sys.executable, d + "/feedback.py", "--artifact", d, "--task", d + "/skill_task.json", "--ann", work + "/ann.jsonl"], capture_output=True, text=True); assert o.returncode == 0, o.stderr[-600:]
    ta = answer(d, teach); nb = answer(d, neigh); bysrc = collections.defaultdict(list)
    for r, x in zip(teach, ta): bysrc[r["src"]].append(x == "Recipe")
    res[arm] = dict(stdout=o.stdout.strip(), tree_identical_to_base=sha(d) == before, patch_bytes=os.path.getsize(d + "/patch.pt") if os.path.exists(d + "/patch.pt") else 0,
                    teach=sum(x == "Recipe" for x in ta) / len(ta), teach_by_src={k: [round(sum(v) / len(v), 3), len(v)] for k, v in bysrc.items()}, neigh_kept=sum(x == r["kind"] for x, r in zip(nb, neigh)) / len(neigh), n_neigh=len(neigh),
                    off_menu_samples=collections.Counter(x for x in ta if x not in OLD).most_common(3))
    print(arm, json.dumps(res[arm]), flush=True); json.dump(res, open(out, "w"), indent=1)
    if arm == "real": shutil.copy(d + "/patch.pt", work + "/patch_real.pt"); res["bytes_acct_real"] = {f: os.path.getsize(d + "/" + f) for f in sorted(os.listdir(d))}; res["total_bytes_real"] = sum(os.path.getsize(d + "/" + f) for f in os.listdir(d))
    else: shutil.rmtree(d)
json.dump(res, open(out, "w"), indent=1)
