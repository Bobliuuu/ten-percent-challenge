#!/usr/bin/env python
"""Harness lock for the held-out exam, written BEFORE any build result is final. Read-only LOCK_HARNESS.json, never overwritten.
Binds: exam file sha256 (hashed as raw bytes, rows not parsed/printed), the task LOCK.json it must match, base id + FP16
bytes, exact byte budget, pre-registered margin/alpha, harness file hashes, timestamp. Evaluates no model.
Usage: lock.py --exam-file F --task-lock v2/base/LOCK.json --split locked --base-id ID --base-fp16-bytes N --out OUT [--standin]"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C, stats as S

ap = argparse.ArgumentParser()
ap.add_argument("--exam-file", required=True); ap.add_argument("--task-lock", required=True); ap.add_argument("--split", required=True)
ap.add_argument("--task", default="mmlu", choices=["mmlu", "mdc6", "c2"]); ap.add_argument("--base-id", required=True); ap.add_argument("--base-revision"); ap.add_argument("--base-fp16-bytes", type=int, required=True)
ap.add_argument("--margin", type=float, default=S.MARGIN); ap.add_argument("--alpha", type=float, default=S.ALPHA)
ap.add_argument("--budget-fraction", type=float, default=C.BUDGET_FRACTION); ap.add_argument("--out", required=True); ap.add_argument("--standin", action="store_true")
a = ap.parse_args()
if os.path.exists(a.out): sys.exit(f"REFUSING: {a.out} exists (a lock is never overwritten).")
if not a.standin and (a.budget_fraction != C.BUDGET_FRACTION or not a.split.endswith("locked")): sys.exit("real lock: split=locked (or C2/locked), fraction 0.10")
TL = json.load(open(a.task_lock))
if "splits" in TL: T = TL["splits"][a.split]
else: T = dict(n=TL["n"][a.split], file_sha256=TL["file_sha256"][a.split])   # v2/alt/LOCK.json layout
fsha = C.sha256_file(a.exam_file)
if fsha != T["file_sha256"]: sys.exit("exam file sha256 does not match task LOCK.json")
if not a.standin and a.base_fp16_bytes <= 2 * 500_000_000: sys.exit("rule 1: base must be > 0.5B params")
lock = dict(standin=a.standin, split=a.split, n_rows=T["n"], exam_file_sha256=fsha, exam_content_sha256=T.get("content_sha256"), task=a.task,
            task_lock_sha256=C.sha256_file(a.task_lock), base_id=a.base_id, base_revision=a.base_revision,
            base_fp16_bytes=a.base_fp16_bytes, budget_fraction=a.budget_fraction,
            budget_bytes=a.budget_fraction * a.base_fp16_bytes, margin=a.margin, alpha=a.alpha,
            criterion="exact one-sided non-inferiority on paired correctness; see CRITERION.md",
            harness_sha256=C.harness_hashes(), locked_utc=C.now())
C.write_json_atomic(a.out, lock, ro=True)
print(json.dumps({k: lock[k] for k in ("split", "n_rows", "exam_file_sha256", "budget_bytes", "margin", "locked_utc", "standin")}, indent=2))
