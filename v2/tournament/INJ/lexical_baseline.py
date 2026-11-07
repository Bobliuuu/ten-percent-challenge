"""Triviality check (rule 8 / 'no cheap shortcut'): regex rule, length rule, TF-IDF+LR (trained on pool_S1 only) on DEV sets and the S2 AUDIT slice.
NEVER reads locked_*.jsonl. Reports balanced accuracy per class."""
import json, re, sys, os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
ld = lambda n: [json.loads(l) for l in open(f"{D}/{n}.jsonl")]
assert "locked" not in " ".join(sys.argv)
RX = re.compile(r"ignor(e|ing)\b.{0,40}(instruction|prompt|previous|above|prior|all)|disregard|system prompt|jailbreak|\bDAN\b|developer mode|you are now|pretend (to be|you)|do anything now|forget (all|everything|your)|reveal your|override", re.I | re.S)
def bal(y, p):
    y = np.array(y); p = np.array(p); return [round(float((p[y == c] == c).mean()), 3) for c in (0, 1)] + [round(float(((p[y == 0] == 0).mean() + (p[y == 1] == 1).mean()) / 2), 3)]
sets = {n: ld(n) for n in ["pool_S1", "dev_S1", "dev_S3", "audit_S2"]}
y = lambda rows: [int(r["gold"] == "injection") for r in rows]
tv = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=200000)
Xp = tv.fit_transform([r["text"] for r in sets["pool_S1"]]); lr = LogisticRegression(C=10, max_iter=2000, class_weight="balanced").fit(Xp, y(sets["pool_S1"]))
# length-threshold rule fit on pool_S1
L = np.array([r["nchar"] for r in sets["pool_S1"]]); yy = np.array(y(sets["pool_S1"]))
ths = np.unique(np.percentile(L, np.linspace(1, 99, 99))); best = max(ths, key=lambda t: max(((L > t) == yy).mean(), ((L <= t) == yy).mean()))
sgn = ((L > best) == yy).mean() >= ((L <= best) == yy).mean()
res = {}
for n in ["dev_S1", "dev_S3", "audit_S2"]:
    r = sets[n]; yt = y(r)
    res[n] = dict(n=len(r), regex=bal(yt, [int(bool(RX.search(x["text"]))) for x in r]),
                  length=bal(yt, [int((x["nchar"] > best) == sgn) for x in r]), tfidf_lr=bal(yt, lr.predict(tv.transform([x["text"] for x in r])).tolist()),
                  benign_with_trigger_frac=round(float(np.mean([bool(RX.search(x["text"])) for x in r if x["gold"] == "benign"])), 3))
print(json.dumps(res, indent=1))
json.dump(res, open(os.path.join(os.path.dirname(D), "results", "lexical_baseline.json"), "w"), indent=1)
