"""ONE COMMAND THAT ANSWERS QUESTIONS.
  user mode:     python answer.py --artifact DIR --task task.json [--in items.jsonl|-] [text ...]      (prints one label per line)
  harness mode:  python answer.py --artifact DIR --in view.json --out out.json [--dump-prompts p.json]  (view = [{id, query, tools}], query = the full prompt; out = [{id, text}])
The text is the model's own generated word; uses DIR/patch.pt automatically if feedback was given. No constrained decoding."""
import sys, os, json, argparse
sys.dont_write_bytecode = True   # never write into the artifact dir
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bonsai_loader as BL, common
from transformers import AutoTokenizer
a = argparse.ArgumentParser(); a.add_argument("--artifact", "--dir", dest="art", default=os.path.dirname(os.path.abspath(__file__))); a.add_argument("--task"); a.add_argument("--in", dest="inp"); a.add_argument("--out"); a.add_argument("--dump-prompts"); a.add_argument("texts", nargs="*"); a = a.parse_args()
T = common.task(a.task) if a.task else None; rows = [{"text": t} for t in a.texts]
if a.inp:
    if a.out: rows += json.load(open(a.inp))
    else:
        f = sys.stdin if a.inp == "-" else open(a.inp); rows += [json.loads(l) if l.lstrip().startswith("{") else {"text": l.rstrip("\n")} for l in f if l.strip()]
prompts = [common.prompt_of(r, T) for r in rows]
if a.dump_prompts: json.dump(prompts, open(a.dump_prompts, "w"))
m, _ = BL.load(a.art); tok = AutoTokenizer.from_pretrained(a.art); outs = common.generate(m, tok, prompts)
if a.out: json.dump([{"id": r["id"], "text": o} for r, o in zip(rows, outs)], open(a.out, "w"))
else:
    for o in outs: print(o)
