"""Reporting-only evaluation of the SHIPPED feedback/answer commands on the locked teach + neighbour sets (v2/fb_ablation/exam_data/SKILL7.json), 3 draws x {real, shuffled, wrong} (+ empty).
Model is decoded once from the packed body; each condition calls feedback.main(...) and answer.main(...) (the same functions the CLIs run) with the shared model; draw 0 'real' is also run as true subprocesses
(feedback.py then answer.py, fresh decode each) and compared. usage: python eval_locked.py ARTIFACT OUT.json --cols 1536 [--data exam_data/SKILL7.json] [--draws 3]"""
import sys, os, json, random, argparse, subprocess, tempfile, collections, time, torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import common, feedback, answer
from fb_core import FB, apply_patch, METHOD
ap = argparse.ArgumentParser(); ap.add_argument("artifact"); ap.add_argument("out"); ap.add_argument("--cols", type=int, default=1024); ap.add_argument("--data", default="exam_data/SKILL7.json"); ap.add_argument("--draws", type=int, default=3); ap.add_argument("--no-subprocess", action="store_true")
a = ap.parse_args(); T0 = time.time(); log = lambda *x: print(f"[{time.time()-T0:6.0f}s]", *x, flush=True)
DD = json.load(open(a.data)); NEW = DD["new"]; LABS = DD["labels"]; task = common.task(); P = task["prompt"].split("{t}")[0]
req = lambda r: r["user"][len(P):] if r["user"].startswith(P) else (_ for _ in ()).throw(ValueError("prompt mismatch"))
cal = DD["cal"]; random.Random(0).shuffle(cal); ann = DD["ann_pool"]; teach = DD["teach"]; neigh = DD["neigh"]; shuf_pool = cal[300:700]
pf = common.prompt_fn(task); tok, m = common.load_model(a.artifact, os.path.join(a.artifact, "no_patch_.pt"), log=log)
fb = FB(tok, m, pf, common.calib()); orig = {li: m.model.layers[li].self_attn.o_proj.weight.data.clone() for li in METHOD["layers"]}
def ans(rows, patch=None):
    if patch: apply_patch(m, patch)
    return answer.answer_texts(tok, m, [req(r) for r in rows], pf)
def undo():
    for li, w in orig.items(): m.model.layers[li].self_attn.o_proj.weight.data.copy_(w)
bn = ans(neigh, None); NE = [i for i, r in enumerate(neigh) if bn[i] == r["kind"]]; ab = ans(ann, None); log("base neigh ok", len(NE), "/", len(neigh))
out = dict(cols=a.cols, n_neigh_ok=len(NE), runs=[]); td = tempfile.mkdtemp(); ANN_BASE = ab
def evaluate(cond, d, patch, extra=None):
    gt = ans(teach, patch); gn = ans([neigh[i] for i in NE]); undo()
    rec = dict(cond=cond, draw=d, teach_gen=sum(g == NEW for g in gt) / len(gt), teach_same=sum(g == NEW for g, r in zip(gt, teach) if r["src"] in ("massive", "clinc150")) / max(1, sum(r["src"] in ("massive", "clinc150") for r in teach)),
               teach_cross=sum(g == NEW for g, r in zip(gt, teach) if r["src"] in ("mtop", "hwu64")) / max(1, sum(r["src"] in ("mtop", "hwu64") for r in teach)), neigh_gen=sum(g == neigh[i]["kind"] for g, i in zip(gn, NE)) / len(NE), teach_outputs=dict(collections.Counter(gt)), gt=gt, gn=gn)
    if extra: rec.update(extra)
    log({k: v for k, v in rec.items() if k not in ("gt", "gn")}); out["runs"].append(rec); json.dump(out, open(a.out, "w"), indent=1); return rec
# empty feedback through the CLI: no patch
open(td + "/empty.jsonl", "w").close(); pe = feedback.main(["--artifact", a.artifact, "--ann", td + "/empty.jsonl", "--out", td + "/p.pt"], fb=fb); evaluate("empty", 0, pe)
for d in range(a.draws):
    rr = random.Random(100 * 25 + d); ix = rr.sample(range(len(ann)), 25)
    conds = {"real": [(req(ann[i]), NEW) for i in ix], "wrong": [(req(ann[i]), rr.choice([l for l in LABS if l != ANN_BASE[i]])) for i in ix], "shuffle": [(req(shuf_pool[i]), NEW) for i in rr.sample(range(len(shuf_pool)), 25)]}
    for c in ("real", "shuffle", "wrong"):
        f = f"{td}/ann_{c}_{d}.jsonl"; open(f, "w").write("".join(json.dumps(dict(text=t, label=l)) + "\n" for t, l in conds[c]))
        p = feedback.main(["--artifact", a.artifact, "--ann", f, "--out", f"{td}/patch_{c}_{d}.pt", "--cols", str(a.cols)], fb=fb); rec = evaluate(c, d, p, dict(patch_bytes=os.path.getsize(f"{td}/patch_{c}_{d}.pt")))
        if c == "real" and d == 0: REAL0 = (f, rec)
if not a.no_subprocess:   # true CLI processes (fresh decode each); free this process's weights first
    f, rec = REAL0; del fb; m = None; torch.cuda.empty_cache(); w = f"{td}/art"; os.makedirs(w, exist_ok=True)
    for fn in os.listdir(a.artifact):
        if not os.path.exists(f"{w}/{fn}"): os.symlink(os.path.abspath(f"{a.artifact}/{fn}"), f"{w}/{fn}")
    t = time.time(); subprocess.check_call([sys.executable, HERE + "/feedback.py", "--artifact", w, "--ann", f, "--cols", str(a.cols), "--out", w + "/patch.pt"]); log("CLI feedback secs", round(time.time() - t))
    open(td + "/teach.txt", "w").write("".join(req(r).replace("\n", " ") + "\n" for r in teach))
    t = time.time(); o = subprocess.check_output([sys.executable, HERE + "/answer.py", "--artifact", w, "--in", td + "/teach.txt"], text=True).split("\n")[:len(teach)]; log("CLI answer secs", round(time.time() - t))
    out["cli_subprocess_real_d0"] = dict(teach_gen=sum(x == NEW for x in o) / len(teach), agree_with_inprocess=sum(x == y for x, y in zip(o, rec["gt"])) / len(teach), patch_bytes=os.path.getsize(w + "/patch.pt"))
    log(out["cli_subprocess_real_d0"]); json.dump(out, open(a.out, "w"), indent=1)
log("DONE")
