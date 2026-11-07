"""Shared helpers for the T-REDTEAM suite. READ-ONLY. NEVER opens tournament/*/data/locked.jsonl or T2 *_locked.jsonl
(except tests gated behind REDTEAM_ALLOW_LOCKED=1, which the red team did NOT run)."""
import json, os, sys, re, hashlib, collections
import pytest
V2 = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
T1 = f"{V2}/tournament/T1"; T2 = f"{V2}/tournament/T2"
D1 = f"{T1}/data"; D2 = f"{T2}/data"
sys.path.insert(0, T1); sys.path.insert(0, f"{V2}/harness")
LOCKED_NAMES = ("locked.jsonl", "s1_locked.jsonl", "s2_locked.jsonl")
ALLOW_LOCKED = os.environ.get("REDTEAM_ALLOW_LOCKED") == "1"

def load(path):
    assert os.path.basename(path) not in LOCKED_NAMES or ALLOW_LOCKED, f"refusing to open locked file {path}"
    return [json.loads(l) for l in open(path) if l.strip()]

def norm(s): return re.sub(r"\W+", " ", s.lower()).strip()
def sig(o): return hashlib.md5(json.dumps(o, sort_keys=True).encode()).hexdigest()

@pytest.fixture(scope="session")
def t1(): return {k: load(f"{D1}/{k}.jsonl") for k in ["s1_dev", "s1_calib", "s1_pool", "s3_dev", "s3_pool"]}
@pytest.fixture(scope="session")
def t1lock(): return json.load(open(f"{D1}/LOCK.json"))
