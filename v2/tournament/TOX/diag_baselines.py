"""Triviality check using NON-LOCKED data only (refuses locked.jsonl).  (1) lexicon rule: predict by lexicon group hit; (2) TF-IDF + LR trained on s1_pool (+calib),
evaluated on s1_dev, s3_dev, s2_probe.  Reports per-cell accuracy of each baseline so we can see that no keyword rule solves the task. Counts/accuracies only."""
import json, sys, os, collections, numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
D = os.environ.get("TOX_DATA", os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import scorer, re
LX = {"toxic": r"\b(?:idiot|idiots|stupid|moron|morons|dumb|loser|scum|trash|pathetic|ugly|fool|jerk|liar|shit|fuck\w*|asshole|bitch|bastard|dick|crap)\b",
      "hate": r"\b(?:racist|racism|jews?|muslims?|islam|gays?|blacks?|immigrants?|illegals?|f[a@]gg?ot|n[i1]gg\w+|retard\w*|sexist|homosexual)\b",
      "sexual": r"\b(?:sex|sexual|porn|naked|nude|breasts?|penis|vagina|erotic|prostitute|orgasm|incest|rape)\b",
      "violence": r"\b(?:kill|killed|killing|murder|shoot|shooting|bomb|stab|attack|beat|punch|strangle|behead|die|dead|gun|guns)\b"}
LXR = {k: re.compile(v, re.I) for k, v in LX.items()}
def lex_pred(t):   # priority: violence > sexual > hate > toxic, else safe
    for k in ("violence", "sexual", "hate", "toxic"):
        if LXR[k].search(t): return k
    return "safe"
def load(f): return [json.loads(l) for l in open(f"{D}/{f}.jsonl")]
tr = load("s1_pool") + load("s1_calib"); ev = {"s1_dev": load("s1_dev"), "s3_dev": load("s3_dev"), "s2_probe": load("s2_probe")}
v = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=300000); X = v.fit_transform([r["text"] for r in tr]); y = [r["gold"] for r in tr]
lr = LogisticRegression(max_iter=300, C=3.0).fit(X, y)
out = {}
for name, rows in ev.items():
    P = {"lexicon_rule": [lex_pred(r["text"]) for r in rows], "tfidf_lr_trained_on_S1pool": list(lr.predict(v.transform([r["text"] for r in rows])))}
    for bn, pr in P.items():
        cells = collections.defaultdict(lambda: [0, 0])
        for r, p in zip(rows, pr):
            k = f"{r['source']}/{r['kind']}" + ("/hard" if r["kind"] == "safe" and r["hard"] else "/plain" if r["kind"] == "safe" else "")
            cells[k][0] += p == r["gold"]; cells[k][1] += 1; cells["ALL"][0] += p == r["gold"]; cells["ALL"][1] += 1
        out[f"{name}|{bn}"] = {k: round(a / n, 3) for k, (a, n) in sorted(cells.items())}
print(json.dumps(out, indent=1)); json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "baselines_keyword_lr.json"), "w"), indent=1)
