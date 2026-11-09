"""L10 ANSWER COMMAND: python answer.py --pack body.pack [--task task.json] [--in items.jsonl|-] [text ...]   (one skill word per line, real greedy generation, no constrained decoding)
Decodes the entropy-coded pack (packlib.py, needs a CUDA GPU + `pip install constriction`) into the text-path model in memory and answers. If feedback was given, pass the edited pack (feedback.py output)."""
import sys, os, json, argparse
sys.dont_write_bytecode = True; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common, packlib
a = argparse.ArgumentParser(); a.add_argument("--pack", required=True); a.add_argument("--task", default=os.path.dirname(os.path.abspath(__file__)) + "/task.json"); a.add_argument("--in", dest="inp"); a.add_argument("texts", nargs="*"); a = a.parse_args()
T = common.task(a.task); rows = [{"text": t} for t in a.texts]
if a.inp:
    f = sys.stdin if a.inp == "-" else open(a.inp); rows += [json.loads(l) if l.lstrip().startswith("{") else {"text": l.rstrip("\n")} for l in f if l.strip()]
m, tok, meta = packlib.build_model(a.pack); tok.pad_token = tok.pad_token or tok.eos_token
for o in common.generate(m, tok, [common.prompt_of(r, T) for r in rows]): print(o)
