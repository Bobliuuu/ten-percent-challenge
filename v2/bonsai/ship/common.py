"""Shared helpers: task spec + chat prompt + greedy label generation with the shipped pack."""
import json, torch
def task(path): return json.load(open(path))          # {"prompt": "... {t} ...", "labels": [...]}
def chat(tok, u): return tok.apply_chat_template([{"role": "user", "content": u}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
@torch.no_grad()
def answer(m, tok, T, texts, bs=16, mx=14):
    out = [None] * len(texts); order = sorted(range(len(texts)), key=lambda i: len(texts[i])); tok.padding_side = "left"
    for s in range(0, len(order), bs):
        idx = order[s:s + bs]; enc = tok([chat(tok, T["prompt"].replace("{t}", texts[i])) for i in idx], return_tensors="pt", padding=True, add_special_tokens=False, return_token_type_ids=False).to(m.device)
        g = m.generate(**enc, max_new_tokens=mx, do_sample=False, pad_token_id=tok.pad_token_id)
        for j, i in enumerate(idx): out[i] = tok.decode(g[j, enc.input_ids.shape[1]:], skip_special_tokens=True).strip()
    return out
