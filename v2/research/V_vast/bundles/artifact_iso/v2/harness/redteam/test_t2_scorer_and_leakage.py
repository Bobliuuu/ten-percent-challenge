"""T2 (PII) scorer attacks + non-locked leakage. Never opens s1_locked/s2_locked."""
import json, re, random, collections, importlib.util, sys
import pytest
from conftest import T2, D2, load, norm
spec = importlib.util.spec_from_file_location("t2scorer", f"{T2}/scorer.py"); S = importlib.util.module_from_spec(spec); spec.loader.exec_module(S)
js = lambda l: json.dumps([{"type": t, "text": v} for t, v in l])
def sc(raw, text, gold): return S.score_item(raw, text, [dict(type=t, text=v) for t, v in gold])
T = "Anna Schmidt lives at 12 Main St, Paris. Call 555-0100. Her email is anna@x.org."
G = [("NAME", "Anna Schmidt"), ("ADDRESS", "12 Main St, Paris"), ("PHONE", "555-0100"), ("EMAIL", "anna@x.org")]

def test_gold_ok(): assert sc(js(G), T, G)["correct"]
def test_wrong_type_fails(): assert not sc(js([("ID_NUMBER", "555-0100")] + [g for g in G if g[0] != "PHONE"]), T, G)["correct"]
def test_case_variant_fails(): assert not sc(js([("NAME", "anna schmidt")] + G[1:]), T, G)["correct"]
def test_prose_no_pii_fails_format(): assert not S.score_item("I found no PII.", "hello", [])["correct"]
def test_empty_string_fails(): assert not S.score_item("", "hello", [])["correct"]
def test_leading_empty_array_then_answer_fails(): assert not sc("[] then " + js(G), T, G)["correct"]
def test_subsumed_extra_entry_is_credited_by_design():
    assert sc(js(G + [("NAME", "Anna")]), T, G)["correct"]   # identical redaction result -> OK, documented

def test_FINDING_distinct_people_merge_into_one_span():
    """NAME/ADDRESS gap-merging (<=2 chars of ws/comma) credits ONE entry spanning several different people / two different addresses."""
    t = "Participants: Bob, Eve, Tom."
    one = S.canon([("NAME", "Bob, Eve, Tom")], t); sep = S.canon([("NAME", "Bob"), ("NAME", "Eve"), ("NAME", "Tom")], t)
    assert one == sep      # passes today = scorer treats them as equal (tolerance); reviewer will note it merges identities
    t2 = "Ship to 12 Main St, 99 Oak Ave."
    assert S.canon([("ADDRESS", "12 Main St, 99 Oak Ave")], t2) == S.canon([("ADDRESS", "12 Main St"), ("ADDRESS", "99 Oak Ave")], t2)

def test_FINDING_inconsistent_merge_and_vs_comma():
    t = "Bob and Eve met."
    a = S.canon([("NAME", "Bob and Eve")], t); b = S.canon([("NAME", "Bob"), ("NAME", "Eve")], t)
    assert a != b   # same redaction semantics, different score: scorer is not a true 'what gets redacted' metric. False negatives are symmetric across base/cand.

def test_FINDING_same_string_two_types_collapses():
    """every occurrence of a string is attributed to the pred TYPE; 'Paris' as ADDRESS inside NAME 'Paris Hilton' occurrences handled by substring drop."""
    t = "Paris Hilton visited Paris."
    r = S.canon([("NAME", "Paris Hilton"), ("ADDRESS", "Paris")], t)
    assert r == frozenset({("NAME", "Paris Hilton"), ("ADDRESS", "Paris")})
    # but a gold with ONLY the NAME is satisfied by pred NAME only; pred adding ADDRESS Paris is wrong -> fine.
    assert not sc(js([("NAME", "Paris Hilton"), ("ADDRESS", "Paris")]), t, [("NAME", "Paris Hilton")])["correct"]

def test_edge_strip_hides_punctuation_difference():
    assert sc(js([("NAME", "Anna Schmidt."), G[1], G[2], G[3]]), T, G)["correct"]

def test_think_and_fence_stripped():
    assert sc("<think>x</think>```json\n" + js(G) + "\n```", T, G)["correct"]

def test_hallucinated_value_poison():
    assert not sc(js(G + [("NAME", "Bob")]), T, G)["correct"]

def test_gold_value_not_in_text_is_unsatisfiable():
    """if a locked gold entry does not occur in its text (data bug), NO output can be correct. Count in non-locked splits."""
    bad = tot = 0
    for f in ["s1_dev", "s3_dev"]:
        for r in load(f"{D2}/{f}.jsonl")[:4000]:
            tot += 1; c = S.canon([(g["type"], g["text"]) for g in r["gold"]], r["text"])
            bad += any(s.startswith("<<NOT_IN_TEXT>>") for _, s in c)
    print("gold with value not in text:", bad, "/", tot)
    assert bad == 0

def test_gold_vs_gold_roundtrip_nonlocked():
    bad = tot = 0
    for f in ["s1_dev", "s3_dev"]:
        for r in load(f"{D2}/{f}.jsonl")[:3000]:
            tot += 1; bad += not S.score_item(js([(g["type"], g["text"]) for g in r["gold"]]), r["text"], r["gold"])["correct"]
    print("T2 gold-vs-gold failures", bad, "/", tot)
    assert bad / tot < 0.002

def test_s3_nopii_too_small_for_ood_dev_proxy():
    n = sum(r["kind"] == "nopii" for r in load(f"{D2}/s3_dev.jsonl"))
    print("S3-dev nopii:", n)
    assert n >= 50   # EXPECTED TO FAIL (5)

def test_t2_s1_leakage_dev_vs_pool(capsys):
    dev, pool = load(f"{D2}/s1_dev.jsonl")[:6000], load(f"{D2}/s1_pool.jsonl")
    pq = {norm(r["text"]) for r in pool}; ex = sum(norm(r["text"]) in pq for r in dev) / len(dev)
    pe = {(g["type"], g["text"].lower()) for r in pool for g in r["gold"]}
    tot = hit = 0
    for r in dev:
        for g in r["gold"]: tot += 1; hit += (g["type"], g["text"].lower()) in pe
    print(f"T2 exact text dup {ex:.4f}; dev gold entity strings already in pool {hit/tot:.3f}")
    assert ex < 0.005
    assert hit / tot < 0.5, "entity-string reuse (synthetic names/ids) so high that memorised strings answer the exam"

def test_t2_s3_vs_s1_entity_overlap():
    s1 = {(g["type"], g["text"].lower()) for r in load(f"{D2}/s1_pool.jsonl") for g in r["gold"]}
    tot = hit = 0
    for r in load(f"{D2}/s3_dev.jsonl")[:5000]:
        for g in r["gold"]: tot += 1; hit += (g["type"], g["text"].lower()) in s1
    print("S3 entity strings found in S1 pool:", hit / tot); assert hit / tot < 0.05
