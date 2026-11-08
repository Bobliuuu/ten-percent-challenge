"""Shared helpers for the shipped commands: item -> full prompt, chat rendering, greedy label generation with the shipped pack."""
import json, torch
def task(path): return json.load(open(path))          # {"prompt": "... {t} ...", "labels": [...]}
def eos_id(tok):
    v = tok.get_vocab()
    for t in ("<|im_end|>", "<|eot_id|>", "<end_of_turn>"):
        if t in v: return v[t]
    return tok.eos_token_id
def load(art, patch=True):
    """load the (dequantised-simulation) artifact dir and, if DIR/patch.pt exists, overlay the feedback columns on the existing down_proj tensors"""
    import os
    from transformers import AutoModelForCausalLM
    m = AutoModelForCausalLM.from_pretrained(art, torch_dtype=torch.bfloat16, device_map="cuda").eval(); pf = art + "/patch.pt"
    if patch and os.path.exists(pf):
        P = torch.load(pf)
        for li, cc, v in zip(P["layers"], P["cols"], P["vals"]): w = m.model.layers[li].mlp.down_proj.weight.data; w[:, cc.long().to(w.device)] = v.to(w.device)
    return m
def chat(tok, u):
    try: return tok.apply_chat_template([{"role": "user", "content": u}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError: return tok.apply_chat_template([{"role": "user", "content": u}], tokenize=False, add_generation_prompt=True)
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
