"""L2 feedback command (diffuse closed-form edit; no gradient steps, no head/adapter/side memory/prompt).
usage: python feedback.py ARTIFACT_DIR ANNOTATED.jsonl NEWLABEL [--unfolded DIR]
ANNOTATED.jsonl rows: {"id","src","kind","user": full prompt,"answer": NEWLABEL}  (<= 25 rows; the prompt menu must NOT list NEWLABEL).
Runs fb27s.py (same code as the measured experiments) on the unfolded model, edits the existing MLP down_proj of the last two layers by
a calibration-covariance closed-form solve, and writes ARTIFACT_DIR/feedback.pt = low-rank factors (U,V) of the edit (~4.5 MB). answer.py adds U@V.T at load.
NOTE: this wrapper was NOT run end to end; the measured path is fb27s.py with SAVE_FB (results/fb_Recipe_save.*: file reload reproduces the in-memory edit exactly)."""
import sys, os, json, shutil, subprocess, tempfile
art, ann, new = sys.argv[1], sys.argv[2], sys.argv[3]; here = os.path.dirname(os.path.abspath(__file__)); unf = sys.argv[sys.argv.index("--unfolded") + 1] if "--unfolded" in sys.argv else "_unfolded"
d = tempfile.mkdtemp(); rows = [json.loads(l) for l in open(ann)]; assert len(rows) <= 25
shutil.copy(ann, f"{d}/ann_{new}.jsonl"); shutil.copy(ann, f"{d}/teach_{new}.jsonl"); shutil.copy(f"{here}/data/exam/cal.jsonl", f"{d}/cal.jsonl"); shutil.copy(f"{here}/data/exam/neigh.jsonl", f"{d}/neigh.jsonl")
env = dict(os.environ, FBDIR=d, NEWLAB=new, NS=str(len(rows)), SAVE_FB=os.path.join(art, "feedback.pt"), SHUF="Alarm,Weather,Music,Calendar,Lists,News,Transport")
subprocess.check_call([sys.executable, f"{here}/fb27s.py", unf, os.path.join(d, "out.json"), "62,63", "1.0", "8", "1.0", "1"], env=env)
