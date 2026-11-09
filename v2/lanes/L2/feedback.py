"""L2 feedback command (closed-form edit of EXISTING projection tensors; no gradient steps, no head/adapter/side memory/prompt).
usage: python feedback.py --artifact DIR --ann ann.jsonl [--task skill_task.json] [--unfolded DIR] [--label NEWLABEL] [--carriers o:63,o:59,o:55,o:51] [--margin 64] [--control shuffled|wrong]
ann.jsonl rows (<= 25), the schema shared by L1/L2/L9b (v2/lanes/ANNOTATION_SCHEMA.md): {"text": raw request, "label": correct label}
  (or {"query": full prompt, "label": ...}; legacy aliases user/answer accepted). The label may be a NEW word; the task menu must NOT list it.
Default carrier = self_attn.o_proj of the last 4 full-attention layers (63,59,55,51), margin 16 (chosen by the DEV selection rule on Jukebox/Briefing, results/sel/*, before any Recipe item was scored; see REPORT).
The edit is solved by fb_l2x.py (calibration-covariance-protected Woodbury solve, same code as the measured runs) and written as exact low-rank factors (fp16) to ARTIFACT_DIR/feedback.pt (~0.6 MB for 25 annotations).
--control shuffled = labels permuted among the given annotations; --control wrong = new label replaced by a random old label (control arms). An empty annotation set / no feedback.pt = the start model.
Legacy positional form still works: feedback.py ARTIFACT_DIR ANN.jsonl NEWLABEL."""
import sys, os, json, shutil, subprocess, tempfile, argparse, collections
here = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, here); import annot
ap = argparse.ArgumentParser(); ap.add_argument("pos", nargs="*"); ap.add_argument("--artifact"); ap.add_argument("--ann", "--annotations", dest="ann"); ap.add_argument("--task"); ap.add_argument("--label")
ap.add_argument("--unfolded", default=os.environ.get("L2_UNFOLD", "_unfolded")); ap.add_argument("--carriers", default="o:63,o:59,o:55,o:51"); ap.add_argument("--margin", type=float, default=16.0)
ap.add_argument("--control", choices=["shuffled", "wrong"]); ap.add_argument("--shuffle-labels", action="store_true", help="alias of --control shuffled"); a = ap.parse_args()
art = a.artifact or a.pos[0]; annf = a.ann or a.pos[1]; new = a.label or (a.pos[2] if len(a.pos) > 2 else None)
T = annot.load_task(a.task); rows = annot.load_ann(annf, T); assert len(rows) <= 25, "at most 25 annotated items"
if new is None: new = collections.Counter(r["answer"] for r in rows if r["answer"] not in T.get("labels", [])).most_common(1)[0][0]
arm = {"shuffled": "all_shuf", "wrong": "all_wrong"}.get("shuffled" if a.shuffle_labels else a.control, "all")
d = tempfile.mkdtemp(); w = lambda n, r: open(f"{d}/{n}.jsonl", "w").write("".join(json.dumps(x) + "\n" for x in r))
w("ann", [dict(r, id=f"ann{i}", src="ann", kind=r["answer"]) for i, r in enumerate(rows)]); [w(n, []) for n in ("teach", "neigh", "old")]
shutil.copy(f"{here}/data/exam/cal.jsonl", f"{d}/cal.jsonl"); json.dump({"new": new, "shuf_labels": T["labels"]}, open(f"{d}/DATASET.json", "w"))
job = dict(name="feedback", data=d, carriers=a.carriers.split(","), m=a.margin, arms=[arm], draws=1, noeval=True, save=os.path.join(art, "feedback.pt"), save_arm=arm, store="fp16")
json.dump([job], open(f"{d}/job.json", "w"))
subprocess.check_call([sys.executable, f"{here}/fb_l2x.py", f"{d}/job.json", f"{d}/out.jsonl", "--model", a.unfolded]); print(open(f"{d}/out.jsonl").read())
