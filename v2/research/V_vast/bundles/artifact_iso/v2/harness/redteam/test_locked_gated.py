"""OWNER-ONLY tests that DO read locked rows. Skipped unless REDTEAM_ALLOW_LOCKED=1. The red team did NOT run these (rule: never read locked/S2 contents).
Run them once, by the person who owns the lock, BEFORE the single-use exam run; they only print aggregate numbers, never rows."""
import os, collections, re, json
import pytest
from conftest import D1, D2, ALLOW_LOCKED, load, norm, sig
pytestmark = pytest.mark.skipif(not ALLOW_LOCKED, reason="locked rows are off-limits unless REDTEAM_ALLOW_LOCKED=1")

def names(r): return frozenset(t["name"] for t in r["tools"])

def test_locked_T1_overlap_with_nonlocked():
    L = load(f"{D1}/locked.jsonl"); S2 = [r for r in L if r["source"] == "S2"]; S1 = [r for r in L if r["source"] == "S1"]
    other = {k: load(f"{D1}/{k}.jsonl") for k in ["s1_dev", "s1_calib", "s1_pool", "s3_dev", "s3_pool"]}
    allq = {norm(r["query"]) for v in other.values() for r in v}; alln = {n for v in other.values() for r in v for n in names(r)}
    alls = {sig(r["tools"]) for v in other.values() for r in v}
    for tag, rows in (("S1-locked", S1), ("S2-locked", S2)):
        print(tag, "exact query in other:", sum(norm(r["query"]) in allq for r in rows) / len(rows),
              "| all tool names seen elsewhere:", sum(names(r) <= alln for r in rows) / len(rows), "| identical tool list elsewhere:", sum(sig(r["tools"]) in alls for r in rows) / len(rows))
    assert sum(names(r) <= alln for r in S2) / len(S2) < 0.2, "S2 tool names mostly appear in S1/S3: not out-of-distribution by API"

def test_locked_T1_S2_style_vs_S1_S3():
    from test_t1_ood_style_nocall import feats, ks
    import random
    L = [r for r in load(f"{D1}/locked.jsonl") if r["source"] == "S2"]; S1 = load(f"{D1}/s1_dev.jsonl"); S3 = load(f"{D1}/s3_dev.jsonl")
    F2 = [feats(r) for r in L]
    for tag, ref in (("S1", S1[:1500]), ("S3", S3)):
        F = [feats(r) for r in ref]; print("KS S2 vs", tag, {k: round(ks([a[k] for a in F], [b[k] for b in F2]), 2) for k in F[0]})

def test_locked_T1_gold_satisfiable_and_scorer_roundtrip():
    import scorer
    L = load(f"{D1}/locked.jsonl"); bad = 0
    for r in L:
        if r["gold"] and r.get("gold_alts"):
            # build one admissible output from the first allowed non-empty value
            g = [dict(name=c["name"], arguments={k: next((x for x in v if x != ""), "") for k, v in c["arguments"].items() if any(x != "" for x in v)}) for c in r["gold"]]
            txt = "\n".join("<tool_call>" + json.dumps(c) + "</tool_call>" for c in g)
            bad += not scorer.score_item(txt, r)["correct"]
        elif r["gold"]:
            bad += not scorer.score_item(scorer.gold_text(r), r)["correct"]
    print("locked items unsatisfiable by their own gold:", bad, "/", len(L)); assert bad / len(L) < 0.005

def test_locked_T2_gold_roundtrip_and_overlap():
    import importlib.util
    spec = importlib.util.spec_from_file_location("s", f"{D2}/../scorer.py"); S = importlib.util.module_from_spec(spec); spec.loader.exec_module(S)
    for f in ["s1_locked", "s2_locked"]:
        L = load(f"{D2}/{f}.jsonl"); bad = sum(any(s.startswith("<<NOT_IN_TEXT>>") for _, s in S.canon([(g["type"], g["text"]) for g in r["gold"]], r["text"])) for r in L)
        print(f, "gold with poison literal:", bad, "/", len(L)); assert bad / len(L) < 0.002
