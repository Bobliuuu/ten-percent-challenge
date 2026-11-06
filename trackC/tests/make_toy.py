"""Stand-in packed model: naive-Bayes word lexicon (npz) + json memory of word deltas. Trained on train rows outside dev/feedback."""
import json, os, sys, collections, numpy as np
from datasets import load_dataset
out = sys.argv[1]; os.makedirs(out, exist_ok=True)
sp = json.load(open(os.path.expanduser("~/ten-percent/splits.json")))
tr = load_dataset("nyu-mll/glue", "sst2")["train"]
ex = set(sp["dev"]) | set(sp["feedback"]); idx = [i for i in range(8000) if i not in ex]
pos, neg = collections.Counter(), collections.Counter()
for i in idx:
    r = tr[i]; (pos if r["label"] else neg).update(r["sentence"].lower().split())
vocab = sorted(w for w in set(pos) | set(neg) if pos[w] + neg[w] >= 3)
sc = np.array([np.log((pos[w] + 1) / (neg[w] + 1)) for w in vocab], dtype=np.float16)
np.savez(f"{out}/weights.npz", sc=sc); json.dump(vocab, open(f"{out}/vocab.json", "w"))
open(f"{out}/predict.py", "w").write('''
import json, numpy as np
def load(d, memory):
    v = json.load(open(d + "/vocab.json")); s = np.load(d + "/weights.npz")["sc"].astype(np.float32)
    tab = dict(zip(v, s)); mem = json.load(open(memory)) if memory else {}
    def f(texts):
        return [int(sum(tab.get(w, 0.0) + mem.get(w, 0.0) for w in t.lower().split()) > 0) for t in texts]
    return f
''')
fb = tr.select(sp["feedback"][:300]); m = {}
for r in fb:   # memory = a few feedback-derived word nudges
    for w in r["sentence"].lower().split(): m[w] = m.get(w, 0) + (0.2 if r["label"] else -0.2)
json.dump({}, open(f"{out}/../mem0.json", "w")); json.dump(m, open(f"{out}/../mem1.json", "w"))
