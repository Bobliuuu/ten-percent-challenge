"""Audit of the v2 T1 re-lock using ONLY non-locked splits and LOCK.json (locked rows are never opened here; the build's own audit counted locked-vs-rest overlaps).
Independent re-implementation of the leakage checks (not imported from build_data.py)."""
import collections, json, random, re
import pytest
from conftest import load, norm, sig, D1

names = lambda r: {t["name"] for t in r["tools"]}
SPL = ["s1_dev", "s1_calib", "s1_pool"]

def test_lock_cells_and_totals(t1lock):
    c = t1lock["splits"]["locked"]["counts"]; n = t1lock["splits"]["locked"]["n"]
    assert n >= 2000 and all(v >= 330 for v in c.values()), c
    assert {f"{s}/{k}" for s in ("S1", "S2") for k in ("nocall", "single", "multi")} <= set(c)
    assert sum(v for k, v in c.items() if k.startswith("S2/")) >= 500

def test_lock_strata_documented(t1lock):
    st = t1lock["locked_totals"]["strata"]; print(st)
    assert st.get("S1/seen", 0) == 0 and st["S1/unseen"] == sum(v for k, v in t1lock["splits"]["locked"]["counts"].items() if k.startswith("S1/"))
    assert t1lock["locked_totals"]["name_overlap_with_nonlocked"]["S1_items_with_any_tool_name_in_nonlocked"] == 0

def test_lock_records_pins_and_build_hash(t1lock):
    import hashlib
    assert t1lock["build_data_py_sha256"] == hashlib.sha256(open(f"{D1}/../build_data.py", "rb").read()).hexdigest()
    assert t1lock["params"]["louvain"]["n_families"] >= 30 and all(v["sha256"] for v in t1lock["raw_inputs"].values())

def test_api_names_disjoint_across_s1_splits(t1):
    seen = {k: set().union(*[names(r) for r in t1[k]]) for k in SPL}
    for a in SPL:
        for b in SPL:
            if a < b: assert not (seen[a] & seen[b]), (a, b, len(seen[a] & seen[b]))

def test_tool_schema_and_gold_disjoint_across_s1_splits(t1):
    ts = {k: {sig([t]) for r in t1[k] for t in r["tools"]} for k in SPL}
    gs = {k: {json.dumps([g["name"], g["arguments"]], sort_keys=True) for r in t1[k] for g in r["gold"]} for k in SPL}
    for a in SPL:
        for b in SPL:
            if a < b: assert not (ts[a] & ts[b]) and not (gs[a] & gs[b])

def test_no_queries_shared_across_splits_bruteforce(t1):
    """exact brute force (no df cut-off) on a sample: J>=0.6 between ANY dev/calib/pool item and an item of ANOTHER split must not exist."""
    rows = [(k, set(norm(r["query"]).split())) for k in SPL for r in t1[k]]
    inv = collections.defaultdict(list)
    for i, (_, s) in enumerate(rows):
        for w in s: inv[w].append(i)
    rng = random.Random(0); bad = 0
    for i in rng.sample(range(len(rows)), 400):
        k, s = rows[i]; cand = set()
        for w in s: cand.update(inv[w])
        for j in cand:
            if rows[j][0] != k and len(s & rows[j][1]) / len(s | rows[j][1]) >= 0.6: bad += 1
    assert bad == 0, bad

@pytest.mark.parametrize("split", ["s1_dev", "s1_calib"])
def test_nocall_not_lexically_trivial(t1, split):
    from test_t1_ood_style_nocall import lexical_nocall_accuracy
    for m in (3, 4):
        rec, fpr = lexical_nocall_accuracy(t1[split], m); print(split, m, rec, fpr); assert rec < 0.2
    rec5, fpr5 = lexical_nocall_accuracy(t1[split], 5); assert rec5 < 0.25 and abs(rec5 - fpr5) < 0.15      # no signal even at the weakest cut

def test_split_sizes_usable(t1):
    assert all(len(t1[k]) >= 1000 for k in SPL)
    for k in ("s1_dev", "s1_calib"):
        c = collections.Counter(r["kind"] for r in t1[k]); assert min(c.values()) >= 200, c
        assert len({r["family"] for r in t1[k]}) >= 5

def test_s3_dev_unchanged_vs_archive():
    """S3 dev is the OOD dev proxy and the stored probe's S3 half: it must be byte-identical to the pre-relock file."""
    # rows gain a 'family' field so the file hash differs; the id list (and order) must be identical
    assert [r["id"] for r in load(f"{D1}/s3_dev.jsonl")] == [r["id"] for r in load(f"{D1}/../data_archive_pre_relock_20261106/s3_dev.jsonl")]
