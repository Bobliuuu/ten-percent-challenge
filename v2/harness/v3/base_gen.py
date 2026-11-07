#!/usr/bin/env python
"""Base runner: its OWN process, sees only the candidate-view rows (id, query, tools), never labels. Usage:
  base_gen.py --base Qwen/Qwen3-0.6B --in view.json --out base_out.json [--max-new-tokens N]
Network is allowed here (weights come from the hub / cache); it has no label access by construction (it is only handed the view file)."""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen_common
ap = argparse.ArgumentParser(); ap.add_argument("--base", required=True); ap.add_argument("--in", dest="inp", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--max-new-tokens", type=int, default=512); ap.add_argument("--bs", type=int, default=8)
a = ap.parse_args()
rows = json.load(open(a.inp)); t0 = time.time()
assert all(set(r) <= {"id", "query", "tools"} for r in rows), "base got non-view fields"
txt = gen_common.generate(a.base, rows, a.max_new_tokens, a.bs)
json.dump([dict(id=r["id"], text=t) for r, t in zip(rows, txt)], open(a.out, "w")); print("BASE_DONE", len(rows), round(time.time() - t0, 1), "s")
