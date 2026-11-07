"""base accuracy on the TOOLSEL dev exam (data/toolsel_dev.jsonl, all 3397 items) with real greedy generation + strict scorer. python eval_base.py MODEL OUT.json"""
import json, sys, os, collections, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from toolsel_scorer import score_toolsel
from transformers import AutoTokenizer, AutoModelForCausalLM
m = sys.argv[1]; tok = AutoTokenizer.from_pretrained(m); tok.padding_side = "left"
lm = AutoModelForCausalLM.from_pretrained(m, dtype=torch.bfloat16).cuda().eval()
R = [json.loads(l) for l in open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data/toolsel_dev.jsonl"))]
def chat(u):
    try: return tok.apply_chat_template([{"role": "user", "content": u}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError: return tok.apply_chat_template([{"role": "user", "content": u}], tokenize=False, add_generation_prompt=True)
P = [chat(r["user"]) for r in R]; order = sorted(range(len(P)), key=lambda i: len(P[i])); out = [None] * len(P)
with torch.no_grad():
    for s in range(0, len(order), 32):
        idx = order[s:s + 32]; enc = tok([P[i] for i in idx], return_tensors="pt", padding=True).to("cuda")
        o = lm.generate(**enc, max_new_tokens=32, do_sample=False, pad_token_id=tok.pad_token_id)
        for j, i in enumerate(idx): out[i] = tok.decode(o[j, enc.input_ids.shape[1]:], skip_special_tokens=False)
c = collections.defaultdict(lambda: [0, 0]); rows = []
for r, t in zip(R, out):
    sc = score_toolsel(t, r); rows.append(dict(id=r["id"], src=r["src"], kind=r["kind"], correct=sc["correct"], pred=sc["pred"][:80], nmenu=len(r["menu"])))
    for k in ("ALL", r["src"] + "/" + r["kind"], "kind:" + r["kind"]): c[k][0] += sc["correct"]; c[k][1] += 1
print(m, {k: f"{v[0]}/{v[1]}={v[0]/v[1]:.3f}" for k, v in sorted(c.items())})
json.dump(dict(model=m, cells={k: v for k, v in c.items()}, rows=rows), open(sys.argv[2], "w"))
