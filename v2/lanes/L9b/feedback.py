"""L9b feedback command. usage: python feedback.py --artifact DIR --ann ann.jsonl [--task skill_task.json] [--cal feedback_cal.jsonl]
ann.jsonl rows (<= 25), schema shared by L1/L2/L9b (v2/lanes/ANNOTATION_SCHEMA.md): {"text": raw request, "label": correct label}
(or {"query": full prompt, "label": ...}; legacy aliases user/answer accepted). The label may be a NEW word (e.g. Recipe).
Legacy positional form still works: feedback.py ARTIFACT_DIR ANN.jsonl [CAL.jsonl].
--cal: format-calibration rows (default ARTIFACT_DIR/feedback_cal.jsonl = the 300 shipped rows, first 300 of data/skill7_cal.jsonl; counted in shipped bytes).
Writes ARTIFACT_DIR/fb.bin + fb.json (replacement columns of the shared-expert down_proj in the last layers; counted in shipped bytes); answer.py applies them automatically."""
import sys, os, json, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import core, fbcore, annot
if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("pos", nargs="*"); ap.add_argument("--artifact"); ap.add_argument("--ann", "--annotations", dest="ann"); ap.add_argument("--task"); ap.add_argument("--cal"); a = ap.parse_args()
    adir = a.artifact or a.pos[0]; annf = a.ann or a.pos[1]; cal_f = a.cal or (a.pos[2] if len(a.pos) > 2 else os.path.join(adir, "feedback_cal.jsonl"))
    T = annot.load_task(a.task); ann = annot.load_ann(annf, T); assert len(ann) <= 25, "at most 25 annotations"
    cal = [json.loads(l) for l in open(cal_f) if l.strip()][:300]
    model, tok, _ = core.load_packed(adir); R = core.Runner(model, tok)
    carrier = fbcore.apply_feedback(model, tok, R, ann, cal); fbcore.save_carrier(adir, carrier, dict(n_ann=len(ann)))
    print("feedback written:", {l: len(v[0]) for l, v in carrier.items()}, "bytes", os.path.getsize(os.path.join(adir, "fb.bin")))
