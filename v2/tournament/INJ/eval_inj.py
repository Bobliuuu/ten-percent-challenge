"""Plain-HF INJ evaluator (greedy, chat template, fixed prompt) for a base or a (fake-quantised / dequantised) model directory -> per-item json in the same format as probe_base.py.
Use when you hold an HF-loadable model dir (e.g. dense.py/dense_stream.py exported via export_hf.py). For streaming QLinear artifacts use dense_stream.py with compress_inputs/ instead
(its t1 rows `{id, correct, output}` are accepted by retention.py).
  python eval_inj.py --model /path/or/hf-id --out results/cand.json [--files dev_S1,dev_S3] [--bs 16]
Locked files are refused unless --allow_locked (the lock owner's single-use run)."""
import argparse, json, os, sys
H = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, H)
import scorer, prompt
ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True); ap.add_argument("--out", required=True); ap.add_argument("--files", default="dev_S1,dev_S3")
ap.add_argument("--bs", type=int, default=16); ap.add_argument("--data", default=H + "/data"); ap.add_argument("--allow_locked", action="store_true")
a = ap.parse_args()
if "locked" in a.files and not a.allow_locked: sys.exit("refusing to read locked files")
import torch, numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
tok = AutoTokenizer.from_pretrained(a.model); tok.padding_side = "left"; tok.pad_token = tok.pad_token or tok.eos_token
m = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16, device_map="cuda" if torch.cuda.is_available() else None).eval()
items = []
for f in a.files.split(","): items += [json.loads(l) for l in open(f"{a.data}/{f}.jsonl")]
def chat(t):
    msg = [{"role": "user", "content": prompt.user_message(t)}]
    try: return tok.apply_chat_template(msg, add_generation_prompt=True, tokenize=False, enable_thinking=False)
    except TypeError: return tok.apply_chat_template(msg, add_generation_prompt=True, tokenize=False)
P = [chat(r["text"]) for r in items]; order = sorted(range(len(P)), key=lambda i: -len(P[i])); out = [None] * len(P); mar = [None] * len(P)
ids_inj = sorted({i for t, i in tok.get_vocab().items() if t.lstrip("Ġ▁ ").lower().startswith("inj")}); ids_ben = sorted({i for t, i in tok.get_vocab().items() if t.lstrip("Ġ▁ ").lower().startswith("ben")})
for b in range(0, len(P), a.bs):
    idx = order[b:b + a.bs]; enc = tok([P[i] for i in idx], return_tensors="pt", padding=True, add_special_tokens=False).to(m.device)
    with torch.no_grad(): g = m.generate(**enc, max_new_tokens=16, do_sample=False, temperature=None, top_p=None, top_k=None, pad_token_id=tok.pad_token_id, output_scores=True, return_dict_in_generate=True)
    lp = torch.log_softmax(g.scores[0].float(), -1)
    for j, i in enumerate(idx):
        out[i] = tok.decode(g.sequences[j, enc.input_ids.shape[1]:], skip_special_tokens=True)
        mar[i] = float(torch.logsumexp(lp[j, ids_inj], 0) - torch.logsumexp(lp[j, ids_ben], 0))
rows = []
for r, o, mg in zip(items, out, mar):
    v = scorer.parse_label(o); rows.append(dict(id=r["id"], source=r["source"], subset=r["subset"], kind=r["kind"], gold=r["gold"], out=o, pred=v, correct=v == r["gold"], inj_minus_ben_logit=mg))
res = dict(model=a.model, summary=scorer.summarize(rows), items=rows); os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True); json.dump(res, open(a.out, "w"))
print(json.dumps(res["summary"]))
