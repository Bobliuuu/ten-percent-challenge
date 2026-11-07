#!/usr/bin/env python
"""Build the DEV-ONLY smoke inputs from T1 dev data. NEVER touches data/locked.jsonl (refuses any path named locked*).
  exam   = balanced slice of probe_items.jsonl (S1 + S3; S3 is mapped to 'S2' at run time via --source-map S3=S2 -> exercises the OOD-block code path)
  pools  = single-call S1-dev items not in the exam (content-disjoint): base-WRONG ones -> annotations (<=25) and teach-set (>=25), base-RIGHT ones -> neighbors.
           (the base is run once here to sort items; that is task design, as in REDESIGN.md section 4, not tuning on a locked set)
  make_dev_smoke.py --base Qwen/Qwen3-0.6B --t1 ~/ten-percent/v2/tournament/T1 --out DIR [--per-cell 20] [--pool 400] [--n-ann 12] [--n-teach 30] [--n-nb 30]"""
import argparse, json, os, random, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import common3 as C
ap = argparse.ArgumentParser(); ap.add_argument("--base", required=True); ap.add_argument("--t1", default=C.T1); ap.add_argument("--out", required=True)
ap.add_argument("--per-cell", type=int, default=20); ap.add_argument("--pool", type=int, default=400); ap.add_argument("--n-ann", type=int, default=12)
ap.add_argument("--n-teach", type=int, default=30); ap.add_argument("--n-nb", type=int, default=30); a = ap.parse_args()
assert "locked" not in a.t1.lower().split(os.sep)[-1]
os.makedirs(a.out, exist_ok=True); rng = random.Random(0)
probe = C.load_jsonl(os.path.join(a.t1, "probe_items.jsonl")); exam, cnt = [], {}
for r in probe:
    k = (r["source"], r["kind"])
    if cnt.get(k, 0) < a.per_cell: exam.append(r); cnt[k] = cnt.get(k, 0) + 1
C.dump_jsonl(exam, f"{a.out}/exam.jsonl")
key = lambda r: json.dumps([r["query"], r["tools"]], sort_keys=True)
used = {key(r) for r in probe} | {r["id"] for r in probe}
dev = [r for r in C.load_jsonl(os.path.join(a.t1, "data", "s1_dev.jsonl")) if r["kind"] == "single" and key(r) not in used and r["id"] not in used]
rng.shuffle(dev); pool = dev[:a.pool]
import gen_common
texts = gen_common.generate(a.base, C.candidate_view(pool), 512, 8); S = C.get_scorer()
ok = [bool(S.score_item(t, r)["correct"]) for t, r in zip(texts, pool)]
wrong = [r for r, o in zip(pool, ok) if not o]; right = [r for r, o in zip(pool, ok) if o]
print(f"pool {len(pool)}: base right {len(right)} wrong {len(wrong)}")
assert len(wrong) >= a.n_ann + a.n_teach and len(right) >= a.n_nb, "pool too small for the requested sets; raise --pool"
C.dump_jsonl(wrong[:a.n_ann], f"{a.out}/ann.jsonl"); C.dump_jsonl(wrong[a.n_ann:a.n_ann + a.n_teach], f"{a.out}/teach.jsonl"); C.dump_jsonl(right[:a.n_nb], f"{a.out}/nb.jsonl")
