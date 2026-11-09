"""build_keep.py OUT_KEEP.json NKEEP [PREFIX]: choose kept vocab ids = prefix ids (< PREFIX) U most frequent ids (>= PREFIX) on a generic multilingual Wikipedia sample, closed under merges, total NKEEP (+33 specials added later).
Corpus: wikimedia/wikipedia 20231101 (first 300 articles x 40 languages, 3000 chars each). No model, no task data."""
import json, sys, os, collections
os.environ.setdefault("HF_HOME", os.path.expanduser("~/hf_cache"))
from datasets import load_dataset
from tokenizers import Tokenizer
from huggingface_hub import snapshot_download
out, NK = sys.argv[1], int(sys.argv[2]); PRE = int(sys.argv[3]) if len(sys.argv) > 3 else 100000
tokp = snapshot_download("Qwen/Qwen3.8-27B", allow_patterns=["tokenizer.json"]) + "/tokenizer.json"; T = Tokenizer.from_file(tokp)
tj = json.load(open(tokp)); vocab = tj["model"]["vocab"]; inv = {i: t for t, i in vocab.items()}
langs = "en zh es fr de ru ja ar hi bn th te ta ko vi id tr fa he uk pl nl it pt sw ur mr gu kn ml pa my km lo ne si am ka hy kk".split()
cnt = collections.Counter(); nd = 0
for l in langs:
    try:
        ds = load_dataset("wikimedia/wikipedia", f"20231101.{l}", split="train", streaming=True)
        for k, r in enumerate(ds):
            if k >= 300: break
            for i in T.encode(r["text"][:3000]).ids: cnt[i] += 1
            nd += 1
    except Exception as e: print("skip", l, str(e)[:80], flush=True)
print("docs", nd, flush=True)
keep = set(range(min(PRE, 248044)))
merges = tj["model"]["merges"]; parents = {}
for mg in merges:
    a, b = (mg if isinstance(mg, list) else mg.split(" ")); parents[vocab[a + b]] = (vocab[a], vocab[b])
def add_closed(i):
    st = [i]
    while st:
        x = st.pop()
        if x in keep: continue
        keep.add(x)
        if x in parents: st.extend(parents[x])
for i, c in cnt.most_common():
    if len(keep) >= NK: break
    if i < 248044 and i >= PRE: add_closed(i)
keep = sorted(keep); json.dump(keep, open(out, "w")); print("kept", len(keep), "covered mass", sum(c for i, c in cnt.items() if i in set(keep)) / sum(cnt.values()))
