"""Leakage / contamination attacks on the T1 tool-use data. Uses ONLY non-locked splits (dev/calib/pool/S3) and LOCK.json counts.
Locked S1 slice is produced by the SAME sha256(query|tools)%7 mechanism as dev/calib, so dev|pool leakage is an unbiased proxy for
locked-S1|pool leakage (before the audit drops). Each test prints its measurement; asserts encode the bar a hostile reviewer will use.
Tests marked xfail(strict) are KNOWN FAILURES = findings (they fail today; flipping to pass means the fix landed)."""
import collections, json, re, pytest
from conftest import norm, sig, load, D1

def names(r): return frozenset(t["name"] for t in r["tools"])
def gold_key(r): return tuple(sorted((g["name"], json.dumps(g["arguments"], sort_keys=True)) for g in r["gold"]))

def test_split_membership_disjoint_ids(t1):
    seen = {}
    for k, rows in t1.items():
        for r in rows:
            assert r["id"] not in seen, (r["id"], k, seen[r["id"]]); seen[r["id"]] = k

def test_exact_query_dups_between_s1_splits(t1, capsys):
    """audit (build_data.py) only compared LOCKED vs others; dev/calib/pool were never cross-deduped."""
    pool = {norm(r["query"]) for r in t1["s1_pool"]}
    out = {}
    for k in ["s1_dev", "s1_calib"]:
        out[k] = sum(norm(r["query"]) in pool for r in t1[k]) / len(t1[k])
    print("exact-query dup rate vs pool", out)
    # the audit dropped 338 exact + 760 near of ~8.3k S1 locked candidates (13%) -> same rate here means the splits are NOT independent
    assert max(out.values()) < 0.01, f"dev/calib leak into pool by exact query: {out}"  # EXPECTED TO FAIL (about 3.4-3.9%)

def test_toolset_identical_in_pool(t1, capsys):
    """xLAM is template-generated: the same tool list is reused for many queries. Fraction of dev items whose ENTIRE tool list appears verbatim in pool."""
    pool = {sig(r["tools"]) for r in t1["s1_pool"]}
    fr = sum(sig(r["tools"]) in pool for r in t1["s1_dev"]) / len(t1["s1_dev"])
    print("dev items with byte-identical tool list in pool:", round(fr, 3))
    assert fr < 0.5, f"{fr:.1%} of S1-dev tool lists are verbatim in the build pool: S1 cannot show generalisation to unseen APIs"  # EXPECTED TO FAIL (~86%)

def test_every_tool_name_seen(t1):
    pool = {n for r in t1["s1_pool"] for n in names(r)}
    unseen = [r for r in t1["s1_dev"] if names(r) - pool]
    fr = len(unseen) / len(t1["s1_dev"])
    print("dev items with >=1 tool name never in pool:", round(fr, 4), len(unseen))
    assert fr > 0.2, f"only {fr:.1%} of S1-dev items involve any API unseen in the pool: there is no 'unseen tools' stratum"  # EXPECTED TO FAIL (~0.5%)

def test_gold_call_seen_verbatim(t1):
    pool = {gold_key(r) for r in t1["s1_pool"] if r["gold"]}
    sub = [r for r in t1["s1_dev"] if r["gold"]]
    fr = sum(gold_key(r) in pool for r in sub) / len(sub)
    gpool = {(g["name"], json.dumps(g["arguments"], sort_keys=True)) for r in t1["s1_pool"] for g in r["gold"]}
    fr_call = sum(all((g["name"], json.dumps(g["arguments"], sort_keys=True)) in gpool for g in r["gold"]) for r in sub) / len(sub)
    print("dev items whose full gold call-set is in pool:", round(fr, 3), "| every individual call (name+args) in pool:", round(fr_call, 3))
    assert fr_call < 0.10, f"{fr_call:.1%} of dev gold answers (each call name+args) were literally seen in the pool"  # EXPECTED TO FAIL (~24%)

def test_near_dup_recall_of_build_audit(t1):
    """re-implements build_data.near() (inverted index, skip words with df>=300, top-40 candidates, Jaccard>=.8) and compares with brute force on a
    sample of dev queries vs pool: how many near-dups does the audit method MISS?"""
    import random
    rows = t1["s1_pool"]
    T = [set(norm(r["query"]).split()) for r in rows]
    inv = collections.defaultdict(list)
    for i, t in enumerate(T):
        for w in t: inv[w].append(i)
    def audit_hit(t):
        c = collections.Counter()
        for w in t:
            if len(inv[w]) < 300:
                for i in inv[w]: c[i] += 1
        return any(len(t & T[i]) / max(1, len(t | T[i])) >= 0.8 for i, _ in c.most_common(40))
    def brute_hit(t):
        # candidate set via ANY shared rare-ish word (df<5000) then exact Jaccard: the ground truth up to very common-word-only matches
        cand = set()
        for w in t:
            if len(inv[w]) < 5000: cand.update(inv[w])
        return any(len(t & T[i]) / max(1, len(t | T[i])) >= 0.8 for i in cand)
    rng = random.Random(0); sample = rng.sample(t1["s1_dev"], 400)
    a = b = 0
    for r in sample:
        t = set(norm(r["query"]).split())
        ah, bh = audit_hit(t), brute_hit(t); a += ah; b += bh
    print(f"near-dup (J>=0.8) found by audit method: {a}/400, by brute force: {b}/400")
    assert b == 0 or a / b >= 0.95, f"audit recall {a}/{b}"

def test_near_dup_rate_looser_threshold(t1):
    """J>=0.6 (paraphrase family). The audit uses 0.8 only; xLAM paraphrase families leak at 0.6."""
    rows = t1["s1_pool"]; T = [set(norm(r["query"]).split()) for r in rows]
    inv = collections.defaultdict(list)
    for i, t in enumerate(T):
        for w in t: inv[w].append(i)
    import random; rng = random.Random(1); sample = rng.sample(t1["s1_dev"], 400); hit = 0
    for r in sample:
        t = set(norm(r["query"]).split()); c = collections.Counter()
        for w in t:
            if len(inv[w]) < 800:
                for i in inv[w]: c[i] += 1
        hit += any(len(t & T[i]) / max(1, len(t | T[i])) >= 0.6 for i, _ in c.most_common(60))
    print("dev queries with a J>=0.6 neighbour in pool:", hit, "/400")
    assert hit / 400 < 0.10, f"{hit/4:.1f}% of dev queries have a J>=0.6 paraphrase in the build pool"  # EXPECTED TO FAIL

def test_query_with_different_tools_crosses_split(t1):
    """hash is over query|tools -> the SAME query with another tool list lands in a different split (this is why exact dups exist)."""
    pool = collections.defaultdict(list)
    for r in t1["s1_pool"]: pool[norm(r["query"])].append(r)
    n = same_gold = 0
    for r in t1["s1_dev"]:
        for p in pool.get(norm(r["query"]), []):
            n += 1; same_gold += gold_key(p) == gold_key(r)
    print("dev-pool exact-query pairs", n, "of which identical gold", same_gold)
    assert n == 0 or same_gold / n < 0.2

def test_s3_pool_dev_leak_lower_but_nonzero(t1):
    pool = {norm(r["query"]) for r in t1["s3_pool"]}
    fr = sum(norm(r["query"]) in pool for r in t1["s3_dev"]) / len(t1["s3_dev"])
    tp = {sig(r["tools"]) for r in t1["s3_pool"]}; ft = sum(sig(r["tools"]) in tp for r in t1["s3_dev"]) / len(t1["s3_dev"])
    print("S3 dev exact-query in pool", round(fr, 4), "tool list in pool", round(ft, 4))
    assert fr < 0.01

def test_s1_s3_cross_contamination(t1):
    """S3 is the OOD dev proxy; make sure S1 and S3 do not share queries or tool names (otherwise S3 is not OOD relative to S1)."""
    s1q = {norm(r["query"]) for k in ["s1_dev", "s1_calib", "s1_pool"] for r in t1[k]}
    s3q = {norm(r["query"]) for k in ["s3_dev", "s3_pool"] for r in t1[k]}
    s1n = {n for k in ["s1_dev", "s1_calib", "s1_pool"] for r in t1[k] for n in names(r)}
    s3n = {n for k in ["s3_dev", "s3_pool"] for r in t1[k] for n in names(r)}
    print("shared queries", len(s1q & s3q), "shared tool names", len(s1n & s3n), "of S3 names", len(s3n))
    assert len(s1q & s3q) == 0 and len(s1n & s3n) / len(s3n) < 0.05

def test_calibration_files_vs_probe_items():
    """compress/dense/t1/calib_t1.jsonl (what compression is calibrated on) must be disjoint from probe_items.jsonl (what retention is measured on)."""
    from conftest import V2
    C = load(f"{V2}/compress/dense/t1/calib_t1.jsonl"); P = load(f"{V2}/compress/dense/t1/probe_items.jsonl")
    assert not ({r["id"] for r in C} & {r["id"] for r in P})
    assert not ({norm(r["query"]) for r in C} & {norm(r["query"]) for r in P})
    pn = {sig(r["tools"]) for r in P}
    shared = sum(sig(r["tools"]) in pn for r in C) / len(C)
    print("calibration items sharing an identical tool list with a probe item:", round(shared, 3))
    # tool-list sharing is expected in xLAM; reported not asserted. But calibration must be S1 only OR declared:
    srcs = collections.Counter(r["source"] for r in C); print("calib sources", dict(srcs))
    assert set(srcs) == {"S1"}, "calibration uses non-S1 data; README says S3 is dev-only/OOD proxy"

def test_probe_items_are_dev_only(t1):
    """The 600-item probe (probe_items.jsonl, tournament + compress/dense/t1) is a LEGACY dev probe built from the PRE-relock splits: S3 items must still be in s3_dev,
    S1 items must come from the archived pre-relock s1_dev (data_archive_pre_relock_20261106). It must never contain a locked id."""
    from conftest import V2, T1
    P = load(f"{V2}/compress/dense/t1/probe_items.jsonl")
    old_dev = {r["id"] for r in load(f"{T1}/data_archive_pre_relock_20261106/s1_dev.jsonl")}
    s3_dev = {r["id"] for r in t1["s3_dev"]}
    assert all((r["id"] in s3_dev) if r["source"] == "S3" else (r["id"] in old_dev) for r in P)
    new_ids = {r["id"] for k in ("s1_dev", "s1_calib", "s1_pool") for r in t1[k]}
    print("legacy S1 probe items that also exist (same id) in the new splits:", sum(r["id"] in new_ids for r in P if r["source"] == "S1"), "of", sum(r["source"] == "S1" for r in P))
