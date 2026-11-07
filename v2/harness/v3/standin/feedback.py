#!/usr/bin/env python
"""STAND-IN `feedback` entry point: closed-form in-place edit of the output-embedding rows (the lm_head method of v2/feedback_v3/fb3.py,
mode 'mean'). Single forward passes only; the weights are never trained. Contract:
   python feedback.py --artifact DIR --annotations ann.jsonl [--alpha 0.1]
   ann.jsonl rows = {id, query, tools, gold:[{name, arguments}]}  (gold = the correct call(s); may be empty list = a no-call correction, skipped)
Edits the existing tensor in place (same name, shape, dtype, file). Zero annotations or zero error sites -> the files are NOT touched."""
import argparse, json, os, sys
ap = argparse.ArgumentParser(); ap.add_argument("--artifact", required=True); ap.add_argument("--annotations", required=True)
ap.add_argument("--alpha", type=float, default=0.1); ap.add_argument("--device")
a = ap.parse_args()
anns = [json.loads(l) for l in open(a.annotations) if l.strip()]
if not anns: print("FEEDBACK_NOOP no annotations"); sys.exit(0)
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from safetensors.torch import load_file, save_file
#@@GEN_COMMON@@
GENERIC = ["Hello, how are you today?", "Explain what a hash table is.", "Write a short poem about autumn.", "What is the capital of France?",
           "Summarise the plot of Romeo and Juliet in two sentences.", "How do I reverse a list in Python?", "Give me three tips for better sleep.",
           "Translate 'good morning' into Spanish.", "What causes rainbows?", "List the planets of the solar system."]
tok = AutoTokenizer.from_pretrained(a.artifact); dev = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
lm = AutoModelForCausalLM.from_pretrained(a.artifact, torch_dtype=torch.bfloat16).to(dev).eval()
for p in lm.parameters(): p.requires_grad_(False)
def gold_text(g): return "\n".join("<tool_call>\n" + json.dumps(dict(name=x["name"], arguments=x["arguments"])) + "\n</tool_call>" for x in g)
sites = []
with torch.no_grad():
    for it in anns:
        if not it.get("gold"): continue
        p = tok(render(tok, it), add_special_tokens=False).input_ids; g = tok(gold_text(it["gold"]) + "<|im_end|>", add_special_tokens=False).input_ids
        ids = torch.tensor([p + g], device=dev); out = lm.model(ids).last_hidden_state[0].float(); lg = lm.lm_head(out.to(lm.lm_head.weight.dtype)).float()
        for j, t in enumerate(g):
            pos = len(p) + j - 1
            if int(lg[pos].argmax()) != t: sites.append((t, out[pos].clone()))
    if not sites: print("FEEDBACK_NOOP model already right on every annotation token"); sys.exit(0)
    mu = torch.cat([lm.model(tok(tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True), return_tensors="pt", add_special_tokens=False).input_ids.to(dev)).last_hidden_state[0].float() for q in GENERIC]).mean(0)
    by = {}
    for t, h in sites: by.setdefault(t, []).append(h)
    W = lm.lm_head.weight.data; edits = {}
    for t, hs in by.items():
        d = torch.stack(hs).mean(0) - mu; d = d / d.norm()
        edits[t] = (W[t].float() + a.alpha * W[t].float().norm() * d).to(W.dtype).cpu()
# write back into the SAME safetensors tensor (tied models: the shared embedding tensor)
files = sorted(f for f in os.listdir(a.artifact) if f.endswith(".safetensors")); done = False
for f in files:
    path = os.path.join(a.artifact, f); T = load_file(path)
    k = "lm_head.weight" if "lm_head.weight" in T else ("model.embed_tokens.weight" if lm.config.tie_word_embeddings and "model.embed_tokens.weight" in T else None)
    if k is None: continue
    for t, row in edits.items(): T[k][t] = row
    save_file(T, path + ".tmp", metadata={"format": "pt"}); os.replace(path + ".tmp", path); done = True
    print("FEEDBACK_DONE rows_edited", len(edits), "sites", len(sites), "tensor", k); break
if not done: sys.exit("no tensor to edit")
