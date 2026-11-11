"""ONE COMMAND THAT TAKES ANNOTATED ITEMS AS FEEDBACK (deletion carrier, L15).
  user mode:     python feedback.py --artifact DIR --task task.json --ann ann.jsonl      rows {"text": ..., "label": <correct label word, may be a NEW word>}
  harness mode:  python feedback.py --artifact DIR --annotations ann.jsonl               rows {id, query (full prompt), tools, gold:[{"name": label}]}
At most 25 rows. Feedback ONLY DELETES existing weights: it learns which entries of the existing MLP gate_proj matrices to set to zero
(Deletion Tuning, arXiv 2611.10866; see deletion.py) and writes DIR/deletions.bin = the delta-coded list of their positions. No value is
stored, no tensor is added, nothing goes into the prompt, the annotations are not stored. Always starts from the original weights.
Zero rows => deletions.bin removed => artifact byte-identical to the base artifact."""
import sys, os, json, argparse
sys.dont_write_bytecode = True
H = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, H); sys.path.insert(0, os.path.join(H, "..", "L1"))
import torch
import deletion as DL

def calib_prompts(art, T):
    p = os.path.join(art, "feedback_calib.jsonl")
    if not os.path.exists(p): p = os.path.join(H, "..", "L1", "feedback_calib.jsonl")
    return [DL._prompt(json.loads(l), T) for l in open(p) if l.strip()]

def main():
    a = argparse.ArgumentParser(); a.add_argument("--artifact", "--dir", dest="art", default=H); a.add_argument("--task"); a.add_argument("--ann", "--annotations", dest="ann", required=True)
    for k, v in DL.DEF.items():
        if k != "target": a.add_argument("--" + k.replace("_", "-"), dest=k, type=type(v), default=v)
    a = a.parse_args(); out = os.path.join(a.art, "deletions.bin"); T = json.load(open(a.task)) if a.task else None
    ann = [json.loads(l) for l in open(a.ann) if l.strip()]; assert len(ann) <= 25, "at most 25 annotated items"
    if os.path.exists(out): os.remove(out)                      # always learn from the ORIGINAL weights
    if not ann: print(json.dumps({"deleted": 0, "bytes": 0})); return
    from transformers import AutoTokenizer
    m = DL.load_model(a.art); tok = AutoTokenizer.from_pretrained(a.art)
    cfg = {k: getattr(a, k) for k in DL.DEF if k != "target"}
    D = DL.learn_deletions(m, tok, ann, T, calib_prompts(a.art, T), cfg, log=lambda h: print(json.dumps(h), file=sys.stderr, flush=True))
    DL.save(D, out); print(json.dumps({"deleted": D["stats"]["n_deleted"], "frac_of_target": D["stats"]["frac_of_target"], "bytes": os.path.getsize(out)}))

if __name__ == "__main__": main()
