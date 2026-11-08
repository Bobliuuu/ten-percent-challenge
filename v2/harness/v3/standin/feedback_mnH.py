#!/usr/bin/env python
"""`feedback` entry point, method mnH (v2/FEEDBACK_LEGAL.md): closed-form, in-place edit of EXISTING lm_head rows. No gradient, no optimizer, no added tensor,
no stored annotation, nothing looked up at answer time, nothing put in the prompt. Contract:
   python feedback.py --artifact DIR --annotations ann.jsonl
   ann.jsonl rows = {id, query, tools, gold:[{name, arguments}]}.  Zero annotations / zero error sites -> NO file is touched.
For every target token t that the model's own teacher-forced argmax gets wrong on an annotation, row t of the output matrix is replaced by
   row_t + M^-1 H^T (H M^-1 H^T + lam I)^-1 g,    H = hidden states at the error sites, g = (best competitor logit - logit of t) + MARGIN,
   M = second moment of the model's own hidden states on the shipped CALIBRATION inputs (feedback_calib.jsonl, counted in the artifact size) and on its
   own greedy continuation of them (the same data a GPTQ-style Hessian uses; no label is involved)."""
import argparse, json, os, sys
ap = argparse.ArgumentParser(); ap.add_argument("--artifact", required=True); ap.add_argument("--annotations", required=True)
ap.add_argument("--margin", type=float, default=8.0); ap.add_argument("--ridge", type=float, default=0.05); ap.add_argument("--device"); ap.add_argument("--max-sites-per-token", type=int, default=64)
a = ap.parse_args()
anns = [json.loads(l) for l in open(a.annotations) if l.strip()]
if not anns: print("FEEDBACK_NOOP no annotations"); sys.exit(0)
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from safetensors.torch import load_file, save_file
#@@GEN_COMMON@@
tok = AutoTokenizer.from_pretrained(a.artifact); tok.padding_side = "left"
if tok.pad_token is None: tok.pad_token = tok.eos_token
dev = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
lm = AutoModelForCausalLM.from_pretrained(a.artifact, torch_dtype=torch.bfloat16).to(dev).eval()
for p in lm.parameters(): p.requires_grad_(False)
EOS = tok.convert_tokens_to_ids("<|im_end|>")
def gold_text(g): return "\n".join("<tool_call>\n" + json.dumps(dict(name=x["name"], arguments=x["arguments"])) + "\n</tool_call>" for x in g)
def hid(ids): return lm.model(torch.tensor([ids], device=dev)).last_hidden_state[0].float()
sites = []
with torch.no_grad():
    for it in anns:
        if not it.get("gold"): continue
        p = tok(render(tok, it), add_special_tokens=False).input_ids; g = tok(gold_text(it["gold"]) + "<|im_end|>", add_special_tokens=False).input_ids
        out = hid(p + g); lg = lm.lm_head(out.to(lm.lm_head.weight.dtype)).float()
        for j, t in enumerate(g):
            pos = len(p) + j - 1
            if int(lg[pos].argmax()) != t and t != EOS: sites.append((t, out[pos].clone()))
    if not sites: print("FEEDBACK_NOOP model already right on every annotation token"); sys.exit(0)
    cal = [json.loads(l) for l in open(os.path.join(a.artifact, "feedback_calib.jsonl")) if l.strip()]
    X = []
    for it in cal:
        p = tok(render(tok, it), add_special_tokens=False).input_ids
        o = lm.generate(torch.tensor([p], device=dev), max_new_tokens=24, do_sample=False, pad_token_id=tok.pad_token_id)[0, len(p):].tolist()
        X.append(hid(p + o)[len(p) - 1:])
    X = torch.cat(X); D = X.shape[1]; M = X.T @ X / len(X); M = M + a.ridge * M.diagonal().mean() * torch.eye(D, device=dev)
    by = {}
    for t, h in sites: by.setdefault(t, []).append(h)
    W = lm.lm_head.weight.data; edits = {}
    for t, hs in by.items():
        Hs = torch.stack(hs[:a.max_sites_per_token]); wt = W[t].float(); bl = (Hs.to(W.dtype) @ W.T).float(); z = Hs @ wt; bl[:, t] = -1e9
        g = (bl.max(1).values - z + a.margin).clamp(min=0); MiH = torch.linalg.solve(M, Hs.T); G = Hs @ MiH
        edits[t] = (wt + MiH @ torch.linalg.solve(G + 1e-3 * G.diagonal().mean() * torch.eye(len(Hs), device=dev), g)).to(W.dtype).cpu()
files = sorted(f for f in os.listdir(a.artifact) if f.endswith(".safetensors")); done = False
for f in files:
    path = os.path.join(a.artifact, f); T = load_file(path)
    k = "lm_head.weight" if "lm_head.weight" in T else ("model.embed_tokens.weight" if lm.config.tie_word_embeddings and "model.embed_tokens.weight" in T else None)
    if k is None: continue
    for t, row in edits.items(): T[k][t] = row
    save_file(T, path + ".tmp", metadata={"format": "pt"}); os.replace(path + ".tmp", path); done = True
    print("FEEDBACK_DONE rows_edited", len(edits), "sites", len(sites), "tensor", k); break
if not done: sys.exit("no tensor to edit")
