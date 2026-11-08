"""L10 ANSWER COMMAND (voice-skill routing, real greedy generation of the skill word, no constrained decoding).
  python answer.py --artifact DIR [--task task.json] [--in items.jsonl|-] [text ...]    (one label per line)
DIR = HF-format directory of the text-path model (L10 simulation: DEQUANTISED weights of the 9.85% entropy-coded GPTQ artifact; the packed file is accounted in MANIFEST.json but not written in this lane).
If DIR/patch.pt exists (feedback was given) the edited down_proj columns are overlaid on the existing tensors."""
import sys, os, json, argparse
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common
from transformers import AutoTokenizer
a = argparse.ArgumentParser(); a.add_argument("--artifact", required=True); a.add_argument("--task", default=os.path.dirname(os.path.abspath(__file__)) + "/task.json"); a.add_argument("--in", dest="inp"); a.add_argument("texts", nargs="*"); a = a.parse_args()
T = common.task(a.task); rows = [{"text": t} for t in a.texts]
if a.inp:
    f = sys.stdin if a.inp == "-" else open(a.inp); rows += [json.loads(l) if l.lstrip().startswith("{") else {"text": l.rstrip("\n")} for l in f if l.strip()]
m = common.load(a.artifact); tok = AutoTokenizer.from_pretrained(a.artifact); tok.pad_token = tok.pad_token or tok.eos_token
for o in common.generate(m, tok, [common.prompt_of(r, T) for r in rows]): print(o)
