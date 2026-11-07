"""TF-IDF+LR baseline trained on the separate 'train' pool (disjoint clusters), evaluated on the same probe_dev items as the LLM.
Two regimes: (a) in-source: trained per source on that source's train pool; (b) cross-source: trained on S2+S3 train, tested on S1 (and S1+S2 -> S3) restricted to shared queues."""
import json, collections, numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from scipy.sparse import hstack
R = "."
L = lambda f: [json.loads(l) for l in open(f)]
tr, pr = L("data/train.jsonl"), L("data/probe_dev.jsonl")
res = {}
def fit(trs):
    v1 = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1); v2 = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True, min_df=2)
    X = hstack([v1.fit_transform([r["text"] for r in trs]), v2.fit_transform([r["text"] for r in trs])]).tocsr()
    m = LogisticRegression(C=20, max_iter=2000).fit(X, [r["queue"] for r in trs])
    return lambda rs: m.predict(hstack([v1.transform([r["text"] for r in rs]), v2.transform([r["text"] for r in rs])]))
def cells(rs, pred):
    c = collections.defaultdict(list)
    for r, p in zip(rs, pred): c[f'{r["source"]}/{r["queue"]}'].append(p == r["queue"])
    return {k: [float(np.mean(v)), len(v)] for k, v in sorted(c.items())}
for s in ("S1", "S2", "S3", "S4"):
    f = fit([r for r in tr if r["source"] == s]); rs = [r for r in pr if r["source"] == s]; p = f(rs)
    res[f"in_source_{s}"] = dict(acc=float(np.mean([a == r["queue"] for a, r in zip(p, rs)])), cells=cells(rs, p))
for tr_src, te, name in (("S2S3", "S1", "S2+S3 -> S1"), ("S1S2", "S3", "S1+S2 -> S3"), ("S1S3", "S2", "S1+S3 -> S2")):
    trs = [r for r in tr if r["source"] in (tr_src[:2], tr_src[2:])]; f = fit(trs); rs = [r for r in pr if r["source"] == te]
    p = f(rs); res["cross_" + name] = dict(acc=float(np.mean([a == r["queue"] for a, r in zip(p, rs)])), cells=cells(rs, p))
json.dump(res, open("results/baseline_tfidf.json", "w"), indent=1)
for k, v in res.items(): print(k, round(v["acc"], 3))
