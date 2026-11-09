"""python feedback.py <artifact_dir> annotations.jsonl [--m 32]   annotations: {"text": request, "label": skill name}, <= 25 lines (marking the model's real mistakes).
Closed-form diffuse edit of EXISTING o_proj tensors of the last layers (no optimiser, no new tensor, no stored feedback): writes edit_o_proj.pt (dense replacement tensors or column patches, counted in MANIFEST)
next to the weights; runtime.load applies it. New labels need no menu entry: label tokens are existing vocabulary and the unmodified answer path generates them. Empty feedback = no edit file.
Solver/layers/margin come from the "fb" entry of task.json (chosen on dev, see REPORT.md); math and solvers in fbl3.py; calibration covariance from the shipped calib_prompts.jsonl."""
import sys, json, os, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import runtime, fbl3
art, ann = sys.argv[1], sys.argv[2]
rows = [json.loads(l) for l in open(ann) if l.strip()]; assert len(rows) <= 25, "at most 25 annotated items"
for f in os.listdir(art):
    if f.startswith("edit_"): os.remove(f"{art}/{f}")           # edits are recomputed from the base artifact + the annotations
if not rows: print("empty feedback: nothing to do"); sys.exit(0)
model, tok, task = runtime.load(art); cfg = dict(layers=list(range(40, 48)), solver="prot", lam=1000.0, awt=10.0, m=32.0); cfg.update(task.get("fb", {}))
if "--m" in sys.argv: cfg["m"] = float(sys.argv[sys.argv.index("--m") + 1])
cal = [json.loads(l) for l in open(f"{art}/calib_prompts.jsonl")]; fb = fbl3.FB(model, tok, task, cal, cfg["layers"])
new = fb.edit(rows, **cfg); torch.save(new, f"{art}/edit_o_proj.pt"); print("wrote", f"{art}/edit_o_proj.pt", fbl3.edit_bytes(new), "bytes")
