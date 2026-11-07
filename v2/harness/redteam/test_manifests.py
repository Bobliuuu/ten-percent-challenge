"""Manifest / lock / count audit. Reads LOCK.json, SHA256SUMS, file hashes of NON-locked files, and counts only. Never opens locked rows."""
import json, hashlib, os, datetime, re, subprocess
import pytest
from conftest import D1, D2, V2

def sha(p): return hashlib.sha256(open(p, "rb").read()).hexdigest()

def test_t1_lock_hashes_match_nonlocked_files(t1lock):
    for k, v in t1lock["splits"].items():
        if k == "locked": continue
        assert sha(f"{D1}/{k}.jsonl") == v["file_sha256"], k

def test_t1_sha256sums_consistent_with_lock(t1lock):
    sums = dict(l.split()[::-1] for l in open(f"{D1}/SHA256SUMS") if l.strip())
    assert sums["LOCK.json"] == sha(f"{D1}/LOCK.json")
    assert sums["locked.jsonl"] == t1lock["splits"]["locked"]["file_sha256"]   # compares two recorded strings; locked.jsonl itself NOT hashed here
    for k, v in t1lock["splits"].items(): assert sums[f"{k}.jsonl"] == v["file_sha256"]

def test_t1_rule3_counts(t1lock):
    c = t1lock["splits"]["locked"]["counts"]; n = t1lock["splits"]["locked"]["n"]
    per = {}
    for k, v in c.items(): per[k.split("/")[1]] = per.get(k.split("/")[1], 0) + v
    s2 = sum(v for k, v in c.items() if k.startswith("S2/"))
    assert n >= 2000 and all(v >= 300 for v in per.values()) and s2 >= 500, (n, per, s2)

def test_t1_each_kind_per_source_margins(t1lock):
    """margin above the 300 floor, per kind AND per source: exactly-300 cells have zero slack if one item is later found invalid."""
    c = t1lock["splits"]["locked"]["counts"]
    tight = {k: v for k, v in c.items() if v < 330}
    print("cells within 10% of the 300 floor:", tight)
    assert not tight, f"zero-slack cells (a single post-hoc invalid item breaks rule 3): {tight}"   # S2/nocall == 300 exactly

def test_t1_per_kind_within_ood_block_meets_floor(t1lock):
    """Reviewer reading of rule 3 'at least 300 for every kind': is the OOD block itself kind-balanced? S2 multi=437, single=500, nocall=300."""
    c = t1lock["splits"]["locked"]["counts"]
    assert all(v >= 300 for k, v in c.items() if k.startswith("S2/"))

def test_t1_ood_fraction_and_kind_imbalance(t1lock):
    c = t1lock["splits"]["locked"]["counts"]
    s2 = sum(v for k, v in c.items() if k.startswith("S2/")); n = t1lock["splits"]["locked"]["n"]
    print("S2 share of locked:", round(s2 / n, 3), " S2 single/multi/nocall:", {k: v for k, v in c.items() if k.startswith("S2/")})
    assert s2 / n >= 0.3

def test_lock_precedes_first_model_result_by_mtime(t1lock):
    lock_t = os.path.getmtime(f"{D1}/LOCK.json")
    for f in os.listdir(f"{V2}/tournament/T1/results"):
        if f.endswith(".json") and f not in ("SUMMARY.json",):
            assert os.path.getmtime(f"{V2}/tournament/T1/results/{f}") > lock_t, f

def test_lock_committed_BEFORE_results_in_git():
    """A reviewer cannot verify 'locked before any model ran' from mtimes. They can from git: the v2 LOCK.json must be in a commit that precedes every result scored on locked rows.
    (The pre-relock LOCK.json shared a commit with the first dev results; that is why T1 was re-locked before any locked scoring.) Pending until the coordinator commits LOCK.json."""
    def commits(args):
        out = subprocess.run(["git", "-C", V2] + args, capture_output=True, text=True).stdout.split("\n"); return [o for o in out if o]
    lock = commits(["log", "--format=%H", "-S", "v2-relock-2026-11-06", "--", "tournament/T1/data/LOCK.json"])
    if not lock: pytest.xfail("v2 LOCK.json not committed yet (coordinator commits LOCK.json + SHA256SUMS alone first)")
    locked_results = commits(["log", "--format=%H", "--diff-filter=A", "--", "tournament/T1/results_locked"])
    assert not (set(lock) & set(locked_results)), "LOCK.json and locked-row results arrive in the same commit"

def test_raw_inputs_pinned(t1lock):
    """build_data.py depends on raw files fetched from mutable mirrors: LOCK.json must record sha256 + HF revision of every raw input."""
    pins = t1lock["raw_inputs"]; assert pins and all(len(v["sha256"]) == 64 and v.get("hf_revision") for v in pins.values()), pins
    for rel, v in pins.items():
        p = f"{V2}/tournament/T1/raw/{rel}"
        if os.path.exists(p): assert sha(p) == v["sha256"], rel

def test_row_data_not_in_git():
    """dev/calib/pool/locked jsonl are gitignored: the repo alone cannot reproduce results; only hashes are public."""
    out = subprocess.run(["git", "-C", V2, "ls-files", "tournament/T1/data"], capture_output=True, text=True).stdout.split()
    print("tracked:", out)
    assert any(x.endswith("s1_dev.jsonl") for x in out), "dev rows not tracked: a reviewer cannot rerun the probe/compression scores"  # EXPECTED TO FAIL

def test_t1_audit_claim_vs_code():
    """v2 audit must cover: exact all-pairs near-dup join, tool schema (named + anonymous), tool names, gold calls, and cross-dedupe of dev/calib/pool."""
    src = open(f"{V2}/tournament/T1/build_data.py").read()
    audit_part = src.split("# ---------------- audit")[1].split("# ---------------- locked selection")[0]
    for needle in ("near_pairs", "sig_named", "sig_anon", "gold_sigs", '("tn"', '("ga"', "RANK"):
        assert needle in audit_part or needle in src, needle
    assert "louvain_communities" in src and "families" in src

def test_t2_lock_counts():
    L = json.load(open(f"{D2}/LOCK.json"))
    kt = L["locked_kind_totals"]; assert L["locked_total"] >= 2000 and all(v >= 300 for v in kt.values())
    sp = L["splits"]; assert sp["s2_locked"]["n"] >= 500
    # tight cells
    s2k = {k: v for k, v in sp["s2_locked"]["kinds"].items()}; print("T2 S2 kinds", s2k)
    s3 = sp["s3_dev"]["kinds"]; print("T2 S3-dev kinds (OOD dev proxy):", s3)
    assert s3["nopii"] >= 50, f"S3 (OOD dev proxy) has only {s3['nopii']} nopii items: per-kind OOD retention for nopii is unmeasurable pre-lock"   # EXPECTED TO FAIL (5)

def test_t2_locked_hash_file_matches_lock():
    sums = dict(l.split()[::-1] for l in open(f"{D2}/LOCKED.sha256") if l.strip())
    L = json.load(open(f"{D2}/LOCK.json"))
    for k in ["s1_locked", "s2_locked"]:
        assert sums[f"data/{k}.jsonl"] == L["splits"][k]["file_sha256"]
