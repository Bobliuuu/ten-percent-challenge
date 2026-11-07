"""Independent integrity audit of data/ (counts only; no row is printed). Re-computes near-duplicate Jaccard>=0.6 between EVERY pair of files with exact sparse products
(no df cut-off, unlike the builder's candidate filter) -- locked rows are read only to compute aggregate counts."""
import json, os, hashlib, collections, stat, numpy as np, pytest
from sklearn.feature_extraction.text import CountVectorizer
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
FILES = ["locked", "s1_dev", "s1_calib", "s1_pool", "s3_dev", "s3_pool", "s2_probe"]
rows = {f: [json.loads(l) for l in open(f"{D}/{f}.jsonl")] for f in FILES}
LOCK = json.load(open(f"{D}/LOCK.json"))
def test_hashes_and_sums():
    sums = dict(l.split()[::-1] for l in open(f"{D}/SHA256SUMS").read().strip().splitlines())
    for f in FILES:
        h = hashlib.sha256(open(f"{D}/{f}.jsonl", "rb").read()).hexdigest()
        assert LOCK["splits"][f]["file_sha256"] == h and sums[f"{f}.jsonl"] == h
def test_locked_readonly(): assert not os.stat(f"{D}/locked.jsonl").st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH)
def test_cells_ge_330():
    c = collections.Counter((r["source"], r["kind"]) for r in rows["locked"])
    assert len(c) == 10 and min(c.values()) >= 330 and len(rows["locked"]) >= 2000
    assert sum(1 for r in rows["locked"] if r["source"] == "S2") >= 500
    for s in ("S1", "S2"): assert sum(v for (a, b), v in c.items() if a == s) >= 1000
def test_sources_roles():
    assert {r["source"] for r in rows["locked"]} == {"S1", "S2"}
    assert {r["source"] for f in ("s1_dev", "s1_calib", "s1_pool") for r in rows[f]} == {"S1"}
    assert {r["source"] for f in ("s3_dev", "s3_pool") for r in rows[f]} == {"S3"} and {r["source"] for r in rows["s2_probe"]} == {"S2"}
def test_labels(): 
    for f in FILES:
        for r in rows[f]: assert r["kind"] == r["gold"] and r["gold"] in ("safe", "toxic", "hate", "sexual", "violence") and 20 <= len(r["text"]) <= 1000
def test_ids_and_texts_unique_across_files():
    ids = collections.Counter(r["id"] for f in FILES for r in rows[f]); assert max(ids.values()) == 1
    ts = collections.Counter(" ".join(r["text"].lower().split()) for f in FILES for r in rows[f]); assert max(ts.values()) == 1
def test_groups_disjoint_across_files():
    g = collections.defaultdict(set)
    for f in FILES:
        for r in rows[f]: g[r["group"]].add(f)
    assert all(len(v) == 1 for v in g.values())
def test_safe_hard_share_ge_40pct_in_every_safe_cell():
    for f in ("locked", "s1_dev", "s3_dev", "s2_probe"):
        for s in {r["source"] for r in rows[f]}:
            sf = [r for r in rows[f] if r["source"] == s and r["kind"] == "safe"]
            assert 0.4 <= np.mean([r["hard"] for r in sf]) <= 0.6, (f, s)
def test_no_neardup_pair_J06_across_files():
    allr = [(f, r) for f in FILES for r in rows[f]]
    cv = CountVectorizer(binary=True, token_pattern=r"(?u)\b\w+\b", dtype=np.float32); X = cv.fit_transform([r["text"] for _, r in allr]).tocsr(); sz = np.asarray(X.sum(1)).ravel()
    fi = np.array([FILES.index(f) for f, _ in allr]); bad = 0; XT = X.T.tocsr()
    for a in range(0, X.shape[0], 1500):
        P = (X[a:a + 1500] @ XT).tocoo(); i, j, v = P.row + a, P.col, P.data
        m = (j > i) & (fi[i] != fi[j]); jac = v[m] / (sz[i[m]] + sz[j[m]] - v[m]); bad += int((jac >= 0.6).sum())
    assert bad == 0, bad
