#!/usr/bin/env python3
"""L4 answer command from the PACKED body alone: python answer.py --body body.l4p --task SKILL7|BANK6 "user text" ...  prints the generated label verbatim (any string; only an optional 'Category:' prefix is stripped)."""
import argparse, sys, os, re
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch, l4fmt, common
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--body", required=True); ap.add_argument("--task", default="SKILL7"); ap.add_argument("text", nargs="+"); a = ap.parse_args()
    lm, tok, rd = l4fmt.load_packed(a.body, log=lambda *x: None)
    for q in a.text:
        ids = torch.tensor([common.prompt_ids(tok, a.task, q)]).cuda()
        with torch.no_grad(): out = lm.generate(input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=8, do_sample=False, pad_token_id=tok.pad_token_id)
        print(re.sub(r"^\s*category\s*:\s*", "", tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip(), flags=re.I))
if __name__ == "__main__": main()
