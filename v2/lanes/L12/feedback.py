"""L12 feedback command. usage: python feedback.py --artifact DIR --ann ann.jsonl [--task skill_task.json] [--cal feedback_cal.jsonl]
ann.jsonl rows (<= 25): {"text": raw request, "label": correct label} (or {"query": full prompt, "label": ...}). The label may be a NEW word (e.g. Recipe). An empty/absent --ann removes fb.bin (restores the packed model).
Always starts from the packed weights (a second call REPLACES the previous carrier, it does not stack). Writes ARTIFACT_DIR/fb.bin + fb.json (replacement columns of the shared-expert down_proj of the last 10 layers (dev-selected)); answer.py applies them."""
import sys, os, json, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import annot, l12_runtime as RT
if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("pos", nargs="*"); ap.add_argument("--artifact"); ap.add_argument("--ann", "--annotations", dest="ann"); ap.add_argument("--task"); ap.add_argument("--cal")
    ap.add_argument("--nlay", type=int, default=RT.NLAY_FB); ap.add_argument("--lam", type=float, default=RT.LAM); ap.add_argument("--awt", type=float, default=RT.AWT); ap.add_argument("--m", type=float, default=RT.MARGIN); a = ap.parse_args()
    adir = a.artifact or a.pos[0]; annf = a.ann or (a.pos[1] if len(a.pos) > 1 else None); cal_f = a.cal or os.path.join(adir, "feedback_cal.jsonl")
    T = annot.load_task(a.task); ann = annot.load_ann(annf, T) if annf and os.path.exists(annf) else []; assert len(ann) <= 25, "at most 25 annotations"
    if not ann: RT.clear_carrier(adir); print("no annotations: fb.bin removed, packed model restored"); sys.exit(0)
    cal = [json.loads(l) for l in open(cal_f) if l.strip()][:300]
    lm, tok = RT.load(adir, carrier=False); R = RT.Runner(lm, tok)
    carrier, _ = RT.apply_feedback(R, ann, cal, nlay=a.nlay, lam=a.lam, awt=a.awt, margin=a.m); RT.save_carrier(adir, carrier, dict(n_ann=len(ann)))
    print("feedback written:", {l: len(v[0]) for l, v in carrier.items()}, "bytes", os.path.getsize(os.path.join(adir, "fb.bin")))
