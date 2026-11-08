"""L2 feedback command (diffuse closed-form edit; no gradient steps, no head/adapter/side memory/prompt).
usage: python feedback.py --artifact DIR --ann ann.jsonl [--task skill_task.json] [--unfolded DIR] [--label NEWLABEL] [--shuffle-labels]
ann.jsonl rows (<= 25), the schema shared by L1/L2/L9b (v2/lanes/ANNOTATION_SCHEMA.md): {"text": raw request, "label": correct label}
  (or {"query": full prompt, "label": ...}; legacy aliases user/answer accepted). The label may be a NEW word; the task menu must NOT list it.
Legacy positional form still works: feedback.py ARTIFACT_DIR ANN.jsonl NEWLABEL.
--shuffle-labels = control arm (random old labels). Runs fb27s.py (same code as the measured experiments) on the unfolded model, edits the existing
MLP down_proj of the last two layers by a calibration-covariance closed-form solve, writes ARTIFACT_DIR/feedback.pt (low-rank factors, ~4.5 MB).
NOTE: the run also evaluates teach/neighbour items (data/exam/*), i.e. it generates on a few hundred extra items; it does not change the written edit."""
import sys, os, json, shutil, subprocess, tempfile, argparse, collections
here = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, here); import annot
ap = argparse.ArgumentParser(); ap.add_argument("pos", nargs="*"); ap.add_argument("--artifact"); ap.add_argument("--ann", "--annotations", dest="ann"); ap.add_argument("--task"); ap.add_argument("--label")
ap.add_argument("--unfolded", default="_unfolded"); ap.add_argument("--shuffle-labels", action="store_true"); a = ap.parse_args()
art = a.artifact or a.pos[0]; annf = a.ann or a.pos[1]; new = a.label or (a.pos[2] if len(a.pos) > 2 else None)
T = annot.load_task(a.task); rows = annot.load_ann(annf, T); assert len(rows) <= 25, "at most 25 annotated items"
if new is None: new = collections.Counter(r["answer"] for r in rows if r["answer"] not in T.get("labels", [])).most_common(1)[0][0]
d = tempfile.mkdtemp()
for n in (f"ann_{new}", f"teach_{new}"): open(f"{d}/{n}.jsonl", "w").write("".join(json.dumps(r) + "\n" for r in rows))
shutil.copy(f"{here}/data/exam/cal.jsonl", f"{d}/cal.jsonl"); shutil.copy(f"{here}/data/exam/neigh.jsonl", f"{d}/neigh.jsonl")
env = dict(os.environ, FBDIR=d, NEWLAB=new, NS=str(len(rows)), SAVE_FB=os.path.join(art, "feedback.pt"), ARM="shuffled" if a.shuffle_labels else "real", SHUF=",".join(T["labels"]))
subprocess.check_call([sys.executable, f"{here}/fb27s.py", a.unfolded, os.path.join(d, "out.json"), "62,63", "1.0", "8", "1.0", "1"], env=env)
