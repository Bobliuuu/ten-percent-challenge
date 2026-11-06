#!/usr/bin/env python
"""Lock the exam: record sha256 of the locked split's content + timestamp into LOCK.json.
Hashes text+labels row by row; prints nothing about content; evaluates no model.
Usage: lock.py [--split validation] [--out LOCK.json]   (--split dev is the test stand-in)"""
import argparse, datetime, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

ap = argparse.ArgumentParser()
ap.add_argument("--split", default="validation")
ap.add_argument("--out", default=os.path.join(C.HERE, "LOCK.json"))
a = ap.parse_args()
if os.path.exists(a.out):
    sys.exit(f"REFUSING: {a.out} already exists (a lock is never overwritten).")
rows, d = C.load_split(a.split)
from datasets import load_dataset
train = load_dataset("nyu-mll/glue", "sst2")["train"]
overlap = len(C.sent_hashes([r[2] for r in rows]) & C.sent_hashes(train["sentence"]))  # hashes only
lock = dict(
    split=a.split, n_rows=len(rows), content_sha256=C.rows_sha256(rows),
    content_hash_def="sha256 over rows in order of f'{idx}\\t{label}\\t{sentence}\\n' (utf-8)",
    dataset_fingerprint=getattr(d, "_fingerprint", None),
    n_sentences_also_in_train_by_hash=overlap,
    splits_json_sha256=C.sha256_file(C.SPLITS),
    harness_sha256=C.harness_hashes(),
    locked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    base_model=C.BASE_ID, base_safetensors_sha256=C.sha256_file(C.find_base_safetensors()),
)
C.write_json_atomic(a.out, lock, ro=True)
print(json.dumps({k: lock[k] for k in ("split", "n_rows", "content_sha256", "locked_utc", "n_sentences_also_in_train_by_hash")}, indent=2))
