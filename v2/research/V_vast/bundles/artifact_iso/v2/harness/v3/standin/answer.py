#!/usr/bin/env python
"""STAND-IN `answer` entry point (artifact = the base itself). Contract:
   python answer.py --artifact DIR --in view.json --out out.json [--dump-prompts p.json]
   view.json = [{id, query, tools}], out.json = [{id, text}]   (text = the model's raw generated tool call(s))"""
import argparse, json, os, sys
ap = argparse.ArgumentParser(); ap.add_argument("--artifact", required=True); ap.add_argument("--in", dest="inp", required=True)
ap.add_argument("--out", required=True); ap.add_argument("--dump-prompts"); ap.add_argument("--max-new-tokens", type=int, default=512); ap.add_argument("--bs", type=int, default=8)
a = ap.parse_args()
#@@GEN_COMMON@@
rows = json.load(open(a.inp))
txt = generate(a.artifact, rows, a.max_new_tokens, a.bs, prompts_out=a.dump_prompts)
json.dump([dict(id=r["id"], text=t) for r, t in zip(rows, txt)], open(a.out, "w"))
