"""Chat-template tool prompt + greedy batch generation. Used by base_gen.py (base model) and, copied into answer.py, by the stand-in artifact.
Same function for base and candidate so a prompt-format difference cannot masquerade as a model difference."""
import json


def render(tok, item):
    tools = [{"type": "function", "function": t} for t in item["tools"]]
    msgs = [{"role": "user", "content": item["query"]}]
    try: return tok.apply_chat_template(msgs, tools=tools, add_generation_prompt=True, tokenize=False, enable_thinking=False)
    except TypeError: return tok.apply_chat_template(msgs, tools=tools, add_generation_prompt=True, tokenize=False)


def generate(model_dir, items, max_new_tokens=512, bs=8, device=None, prompts_out=None):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(model_dir); tok.padding_side = "left"
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    lm = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.bfloat16).to(dev).eval()
    prompts = [render(tok, it) for it in items]
    if prompts_out: json.dump(prompts, open(prompts_out, "w"))
    out = [None] * len(items); order = sorted(range(len(items)), key=lambda i: len(prompts[i]))
    with torch.no_grad():
        for s in range(0, len(order), bs):
            idx = order[s:s + bs]
            enc = tok([prompts[i] for i in idx], return_tensors="pt", padding=True, add_special_tokens=False).to(dev)
            g = lm.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tok.pad_token_id)
            for j, i in enumerate(idx): out[i] = tok.decode(g[j, enc.input_ids.shape[1]:], skip_special_tokens=False).replace("<|endoftext|>", "").replace("<|im_end|>", "").strip()
    return out
