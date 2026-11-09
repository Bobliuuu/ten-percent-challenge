"""Give feedback: annotated mistakes (<= 25) -> closed-form edit of existing o_proj tensors (layers 56-63, margin 32, protected by the calibration key covariance,
only the top-`--cols` input columns of each tensor are rewritten) -> patch.pt (the replaced columns + indices; ~84 MB at 1024 columns, ~126 MB at 1536).
  python feedback.py --artifact DIR --ann ann.jsonl [--cols 1536] [--out DIR/patch.pt]     rows: {"text": "how long do I boil an egg", "label": "Recipe"}  (label may be a NEW word)
An empty ann file removes patch.pt (the artifact is then the untouched packed model). No training, no gradient, annotations are not stored. Needs one CUDA GPU >= 80 GB."""
import argparse, json, sys, os, time, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common
from fb_core import FB, METHOD

def main(argv=None, model=None, fb=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--artifact", required=True); ap.add_argument("--task"); ap.add_argument("--ann", required=True); ap.add_argument("--out")
    ap.add_argument("--cols", type=int, default=METHOD["cols"]); ap.add_argument("--calib"); a = ap.parse_args(argv)
    out = a.out or os.path.join(a.artifact, "patch.pt"); rows = [json.loads(l) for l in open(a.ann) if l.strip()]; assert len(rows) <= 25, "at most 25 annotations"
    t0 = time.time()
    if not rows:
        if os.path.exists(out): os.remove(out)
        print("empty feedback: patch removed"); return None
    if fb is None:
        tok, m = model if model else common.load_model(a.artifact, os.path.join(a.artifact, "no_patch_.pt"), log=lambda *x: print(*x, file=sys.stderr))   # feedback is always fitted on the UNpatched model
        fb = FB(tok, m, common.prompt_fn(common.task(a.task)), common.calib(a.calib))
    mt = dict(METHOD, cols=a.cols); patch, n = fb.edit([(r["text"], r["label"]) for r in rows], mt)
    torch.save(patch, out); fb.restore()
    print(json.dumps(dict(nsites=n, patch_bytes=os.path.getsize(out), cols=a.cols, secs=round(time.time() - t0, 1)))); return patch

if __name__ == "__main__": main()
