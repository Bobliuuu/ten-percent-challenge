"""ONE COMMAND THAT ANSWERS QUESTIONS (deletion carrier, L15).
  user mode:     python answer.py --artifact DIR --task task.json [--in items.jsonl|-] [text ...]     (one label per line)
  harness mode:  python answer.py --artifact DIR --in view.json --out out.json                      (view = [{id, query, tools}]; out = [{id, text}])
Loads the model (Bonsai pack via the L1 loader, or an HF folder), zeroes the weights listed in DIR/deletions.bin if feedback was given, and
answers by plain greedy generation (no constrained decoding, no head, no lookup)."""
import sys, os, json, argparse
sys.dont_write_bytecode = True
H = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, H); sys.path.insert(0, os.path.join(H, "..", "L1"))
import deletion as DL, common
from transformers import AutoTokenizer
a = argparse.ArgumentParser(); a.add_argument("--artifact", "--dir", dest="art", default=H); a.add_argument("--task"); a.add_argument("--in", dest="inp"); a.add_argument("--out"); a.add_argument("texts", nargs="*"); a = a.parse_args()
T = json.load(open(a.task)) if a.task else None; rows = [{"text": t} for t in a.texts]
if a.inp:
    if a.out: rows += json.load(open(a.inp))
    else:
        f = sys.stdin if a.inp == "-" else open(a.inp); rows += [json.loads(l) if l.lstrip().startswith("{") else {"text": l.rstrip("\n")} for l in f if l.strip()]
m = DL.load_model(a.art); tok = AutoTokenizer.from_pretrained(a.art)
pf = os.path.join(a.art, "deletions.bin")
if os.path.exists(pf): DL.apply(m, DL.load(pf))
outs = common.generate(m, tok, [common.prompt_of(r, T) for r in rows])
if a.out: json.dump([{"id": r["id"], "text": o} for r, o in zip(rows, outs)], open(a.out, "w"))
else:
    for o in outs: print(o)
