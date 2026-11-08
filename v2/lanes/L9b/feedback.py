"""L9b feedback command. usage: python feedback.py ARTIFACT_DIR ANNOTATIONS.jsonl [CAL.jsonl]
ANNOTATIONS.jsonl: <=25 rows {"user": full prompt (the same router prompt), "answer": correct label}; labels may be a NEW word (e.g. Recipe). CAL.jsonl: format-calibration rows (default: the 200 rows shipped as data/skill7_cal.jsonl if present).
Writes ARTIFACT_DIR/fb.bin + fb.json (replacement columns of the shared-expert down_proj in the last layers; counted in shipped bytes); answer.py applies them automatically."""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import core, fbcore
if __name__ == "__main__":
    adir, annf = sys.argv[1], sys.argv[2]; cal_f = sys.argv[3] if len(sys.argv) > 3 else os.path.join(adir, "feedback_cal.jsonl")
    ann = [json.loads(l) for l in open(annf)]; assert len(ann) <= 25, "at most 25 annotations"; cal = [json.loads(l) for l in open(cal_f)][:300]
    model, tok, _ = core.load_packed(adir); R = core.Runner(model, tok)
    carrier = fbcore.apply_feedback(model, tok, R, ann, cal); fbcore.save_carrier(adir, carrier, dict(n_ann=len(ann)))
    print("feedback written:", {l: len(v[0]) for l, v in carrier.items()}, "bytes", os.path.getsize(os.path.join(adir, "fb.bin")))
