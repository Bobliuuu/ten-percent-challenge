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
