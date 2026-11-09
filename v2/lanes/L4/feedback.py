#!/usr/bin/env python3
"""L4 feedback command: python feedback.py --body body.l4p --task SKILL7|BANK6 --ann ann.jsonl   (lines {"text":..., "label":...}, <=25 annotations).
Closed-form edit (l4fb.py, L11 o_prot recipe) of the fp16 o_proj carrier of the last 8 layers; the edited tensors are written back INTO the packed file in place (size unchanged). No head, no classifier, nothing in the prompt."""
import argparse, sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch, l4fmt, l4fb, common
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--body", required=True); ap.add_argument("--task", default="SKILL7"); ap.add_argument("--ann", required=True)
    ap.add_argument("--cfg", default="{}", help="json overrides of l4fb.DEF"); a = ap.parse_args()
    lm, tok, rd = l4fmt.load_packed(a.body, log=lambda *x: None)
    cal = [(common.prompt_ids(tok, a.task, r["text"]), r["answer"]) for r in common.cal(a.task)]
    ann = [json.loads(l) for l in open(a.ann) if l.strip()]; assert len(ann) <= 25
    fb = l4fb.FB(lm, tok, cal, json.loads(a.cfg)); n = fb.edit([(common.prompt_ids(tok, a.task, r["text"]), r["label"]) for r in ann])
    for l in fb.LAY: rd.overwrite_raw(f"L{l}.o.raw", lm.model.layers[l].self_attn.o_proj.weight.data)
    print(json.dumps(dict(annotations=len(ann), sites=n, layers_rewritten=len(fb.LAY), body_bytes=os.path.getsize(a.body))))
if __name__ == "__main__": main()
