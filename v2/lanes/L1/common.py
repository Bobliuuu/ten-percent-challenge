"""Shared helpers for the shipped commands: item -> full prompt, chat rendering, greedy label generation with the shipped pack."""
import json, torch
def task(path): return json.load(open(path))          # {"prompt": "... {t} ...", "labels": [...]}
def chat(tok, u): return tok.apply_chat_template([{"role": "user", "content": u}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
def prompt_of(row, T=None):
    """row with "query" = the full prompt already, or "text" = raw input to be put into the task template T."""
    return row["query"] if "query" in row else T["prompt"].replace("{t}", row["text"])
@torch.no_grad()
def generate(m, tok, prompts, bs=16, mx=14):
    out = [None] * len(prompts); order = sorted(range(len(prompts)), key=lambda i: len(prompts[i])); tok.padding_side = "left"
    for s in range(0, len(order), bs):
        idx = order[s:s + bs]; enc = tok([chat(tok, prompts[i]) for i in idx], return_tensors="pt", padding=True, add_special_tokens=False, return_token_type_ids=False).to(m.device)
        g = m.generate(**enc, max_new_tokens=mx, do_sample=False, pad_token_id=tok.pad_token_id)
        for j, i in enumerate(idx): out[i] = tok.decode(g[j, enc.input_ids.shape[1]:], skip_special_tokens=True).strip()
    return out

def _norm_variants(t):
    import re
    return [re.sub(r"[^a-z0-9' ]", "", t.lower()).strip(), t.rstrip(" .?!") + " please", ("hey " + t[:1].lower() + t[1:]) if t else t, t.upper() if len(t) < 40 else t.lower()]
def augment(ann, T, n):
    """annotation augmentation (n extra surface variants per annotated text: lowercase/no punctuation, '... please', 'hey ...', upper/lower). Only for user mode (rows with text). n=0 -> identity."""
    if not n or T is None: return list(ann)
    out = list(ann)
    for r in ann:
        if "text" in r:
            for v in _norm_variants(r["text"])[:n]: out.append({"text": v, "label": r["label"]})
    return out
