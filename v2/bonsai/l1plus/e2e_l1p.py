"""L1+ END-TO-END usability test of the SHIPPED commands (dev data only). python e2e_l1p.py ART WORKDIR EXTRA OUT.json [OLD_SHIP_DIR]
 1. regression: the new feedback.py with default flags writes the SAME patch tensors as the sealed L1 feedback.py (OLD_SHIP_DIR) on the same annotations
 2. persistent vs in-process: `answer.py --feedback ann --save-patch` (in-process feedback + answers) vs feedback.py then answer.py: identical outputs
 3. continual CLI: feedback.py --append x3 (Recipe, Email, Takeaway; 8/8/9 items): teach per skill + neighbours after each round
 4. scale carrier through the CLI (--carrier scale): teach + neighbours + bytes."""
import sys, os, json, subprocess, shutil, random, hashlib, collections
import torch
art, work, extra, out = sys.argv[1:5]; old = sys.argv[5] if len(sys.argv) > 5 else None; HERE = os.path.dirname(os.path.abspath(__file__)); H = lambda s: int(hashlib.sha256(s.encode()).hexdigest()[:8], 16)
rd = lambda p: [json.loads(l) for l in open(p)]; raw = lambda r: r["user"].split("Request: ", 1)[1]; os.makedirs(work, exist_ok=True)
srt = lambda n, t: sorted(rd(f"{extra}/skill_{n}.jsonl"), key=lambda r: H(t + r["id"]))
A = srt("new_a", "fbx-ent|"); EM = srt("email", "l1p-"); TK = srt("takeaway", "l1p-")
items = [r for r in rd(HERE + "/../../task_search/EASY/items_dev.jsonl") if r["task"] == "SKILL7"]
TASKJ = work + "/task.json"; json.dump({"prompt": json.load(open(HERE + "/../../task_search/EASY/PLAN.json"))["SKILL7"]["prompt"].replace("{X}", "{t}"), "labels": ["Alarm", "Weather", "Music", "Calendar", "Lists", "News", "Transport"]}, open(TASKJ, "w"))
def run(cmd): o = subprocess.run([sys.executable] + cmd, capture_output=True, text=True); assert o.returncode == 0, o.stderr[-1000:]; return o.stdout.rstrip("\n")
def answer(d, rows, extra_args=()):
    p = work + "/in.jsonl"; open(p, "w").write("".join(json.dumps({"text": raw(r)}) + "\n" for r in rows)); return run([d + "/answer.py", "--artifact", d, "--task", TASKJ, "--in", p, *extra_args]).split("\n")
def annf(rows, label, name): p = f"{work}/{name}.jsonl"; open(p, "w").write("".join(json.dumps(dict(text=raw(r), label=label)) + "\n" for r in rows)); return p
def fresh(name):
    d = f"{work}/{name}"; shutil.rmtree(d, ignore_errors=True); shutil.copytree(art, d, symlinks=True)
    if os.path.exists(d + "/patch.pt"): os.remove(d + "/patch.pt")
    return d
frac = lambda xs, rows, lab=None: round(sum(x == (lab or r["kind"]) for x, r in zip(xs, rows)) / len(rows), 3)
res = {}; base = answer(art, items); ok = [r for r, o in zip(items, base) if o == r["kind"]]; neigh = ok[::3]; teachA = A[100:]
ann25 = annf(random.Random(25000).sample(A[:60], 25), "Recipe", "ann25")
# 1. regression vs sealed code
d = fresh("new"); res["new_cli"] = json.loads(run([d + "/feedback.py", "--artifact", d, "--task", TASKJ, "--ann", ann25]).split("\n")[-1])
if old:
    d0 = fresh("old"); [shutil.copy(old + "/" + f, d0 + "/" + f) for f in ("feedback.py", "bonsai_loader.py", "common.py", "answer.py")]; res["old_cli"] = json.loads(run([d0 + "/feedback.py", "--artifact", d0, "--task", TASKJ, "--ann", ann25]).split("\n")[-1])
    P0, P1 = torch.load(d0 + "/patch.pt"), torch.load(d + "/patch.pt"); res["regression_patch_identical"] = bool(P0["layers"] == P1["layers"] and all(torch.equal(a, b) for a, b in zip(P0["cols"], P1["cols"])) and all(torch.equal(a, b) for a, b in zip(P0["vals"], P1["vals"]))); res["max_abs_diff"] = float(max((a.float() - b.float()).abs().max() for a, b in zip(P0["vals"], P1["vals"])))
    shutil.rmtree(d0)
o_cli = answer(d, teachA + neigh); res["cli_teach"] = frac(o_cli[:len(teachA)], teachA, "Recipe"); res["cli_neigh"] = frac(o_cli[len(teachA):], neigh)
# 2. in-process
d2 = fresh("inproc"); o_in = answer(d2, teachA + neigh, ["--feedback", ann25]); res["inproc_identical_to_cli"] = o_in == o_cli; res["inproc_leaves_no_patch"] = not os.path.exists(d2 + "/patch.pt")
d3 = fresh("inproc_save"); answer(d3, teachA[:2], ["--feedback", ann25, "--save-patch"]); res["inproc_save_patch_exists"] = os.path.exists(d3 + "/patch.pt"); o_p = answer(d3, teachA + neigh); res["persisted_identical_to_cli"] = o_p == o_cli
shutil.rmtree(d2); shutil.rmtree(d3); shutil.rmtree(d); json.dump(res, open(out, "w"), indent=1); print(json.dumps(res)[:700], flush=True)
# 3. continual
d = fresh("cont"); sets = [("Recipe", A[:60], A[100:]), ("Email", EM[:60], EM[60:200]), ("Takeaway", TK[:50], TK[50:])]; res["continual"] = []
for i, (lab, pool, teach) in enumerate(sets):
    n = 8 if i < 2 else 9; af = annf(random.Random(7 + i).sample(pool, n), lab, f"c{i}"); r = json.loads(run([d + "/feedback.py", "--artifact", d, "--task", TASKJ, "--ann", af, "--append"]).split("\n")[-1])
    step = dict(round=i + 1, label=lab, n=n, patch_bytes=r["patch_bytes"], neigh=frac(answer(d, neigh), neigh))
    for l2, _, t2 in sets[:i + 1]: step["teach_" + l2] = frac(answer(d, t2[:100]), t2[:100], l2)
    res["continual"].append(step); print(json.dumps(step), flush=True); json.dump(res, open(out, "w"), indent=1)
shutil.rmtree(d)
# 4. scale carrier
d = fresh("scale"); r = json.loads(run([d + "/feedback.py", "--artifact", d, "--task", TASKJ, "--ann", ann25, "--carrier", "scale", "--lam", "100"]).split("\n")[-1]); oc = answer(d, teachA + neigh)
res["scale_cli"] = dict(patch_bytes=r["patch_bytes"], teach=frac(oc[:len(teachA)], teachA, "Recipe"), neigh=frac(oc[len(teachA):], neigh)); shutil.rmtree(d); json.dump(res, open(out, "w"), indent=1); print(res["scale_cli"], flush=True)
