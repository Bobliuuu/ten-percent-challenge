"""ONE COMMAND THAT ANSWERS QUESTIONS.   python answer.py --dir <artifact dir> --task tasks/news.json  [--in items.jsonl|-] [text ...]
Reads texts (args, or jsonl lines with a "text" field, or stdin lines), prints one label per line (the model's own generated word). Uses patch.pt automatically if feedback was given."""
import sys, os, json, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bonsai_loader as BL, common
from transformers import AutoTokenizer
a = argparse.ArgumentParser(); a.add_argument("--dir", required=True); a.add_argument("--task", required=True); a.add_argument("--in", dest="inp"); a.add_argument("texts", nargs="*"); a = a.parse_args()
texts = list(a.texts)
if a.inp:
    f = sys.stdin if a.inp == "-" else open(a.inp)
    texts += [json.loads(l)["text"] if l.lstrip().startswith("{") else l.rstrip("\n") for l in f if l.strip()]
m, _ = BL.load(a.dir); tok = AutoTokenizer.from_pretrained(a.dir)
for t, o in zip(texts, common.answer(m, tok, common.task(a.task), texts)): print(o)
