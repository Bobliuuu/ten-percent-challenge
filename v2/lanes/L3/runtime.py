"""L3 runtime: build the sliced Qwen3-30B-A3B MoE from the packed artifact (weights.bin + index.json + config + tokenizer) and generate. No external weights."""
import json, os, sys, math, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import packfmt

def had(n):
    H = torch.ones(1, 1)
    while H.shape[0] < n: H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    return H / math.sqrt(n)
def rot_matrix(dim, blk, seed):
    g = torch.Generator().manual_seed(seed); sg = (torch.randint(0, 2, (dim,), generator=g) * 2 - 1).float(); B = had(blk); Q = torch.zeros(dim, dim)
    for s0 in range(0, dim, blk): Q[s0:s0 + blk, s0:s0 + blk] = B
    return sg[:, None] * Q

def load(art, device="cuda"):
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    idx = json.load(open(f"{art}/index.json")); meta = idx["_meta"]; kept = meta["kept"]; K = len(kept[0]); assert all(len(k) == K for k in kept)
    mm = open(f"{art}/weights.bin", "rb").read(); cfg = AutoConfig.from_pretrained(art); cfg.num_experts = K
    with torch.device("meta"): model = AutoModelForCausalLM.from_config(cfg, torch_dtype=torch.bfloat16)
    T = lambda n: packfmt.read_tensor(mm, idx[n]); sd = {}; bf = torch.bfloat16
    sd["model.embed_tokens.weight"] = T("embed").to(bf); sd["lm_head.weight"] = T("head").to(bf); sd["model.norm.weight"] = T("final_norm").to(bf)
    for l in range(cfg.num_hidden_layers):
        p = f"model.layers.{l}."
        for a, b in (("input_layernorm", "input_layernorm.weight"), ("post_attention_layernorm", "post_attention_layernorm.weight"), ("q_norm", "self_attn.q_norm.weight"), ("k_norm", "self_attn.k_norm.weight")): sd[p + b] = T(f"L{l}.{a}").to(bf)
        for n in ("q_proj", "k_proj", "v_proj", "o_proj"): sd[p + f"self_attn.{n}.weight"] = T(f"L{l}.{n}").to(bf)
        sd[p + "mlp.gate.weight"] = T(f"L{l}.router").to(bf)
        if meta["had"]: Qi = rot_matrix(cfg.hidden_size, min(meta["hblk"], cfg.hidden_size), 1000 + l); Qd = rot_matrix(cfg.moe_intermediate_size, min(meta["hblk"], cfg.moe_intermediate_size), 2000 + l)
        for j, e in enumerate(kept[l]):
            for nm, mod in (("gate", "gate_proj"), ("up", "up_proj"), ("down", "down_proj")):
                w = T(f"L{l}.E{e}.{nm}")
                if meta["had"]: w = w @ (Qd.T if nm == "down" else Qi.T)
                sd[p + f"mlp.experts.{j}.{mod}.weight"] = w.to(bf)
    for f in sorted(os.listdir(art)):   # feedback edits (dense replacement tensors), applied if present
        if f.startswith("edit_") and f.endswith(".pt"): sd.update({k: v.to(bf) for k, v in torch.load(f"{art}/{f}").items()})
    model.load_state_dict(sd, strict=True, assign=True); model = model.to(device).eval()
    return model, AutoTokenizer.from_pretrained(art), json.load(open(f"{art}/task.json"))

@torch.no_grad()
def route(model, tok, task, texts, bs=16, max_new=6):
    out = []
    for i in range(0, len(texts), bs):
        ps = [tok.apply_chat_template([{"role": "user", "content": task["prompt"].replace("{X}", t)}], tokenize=False, add_generation_prompt=True) for t in texts[i:i + bs]]
        tok.padding_side = "left"; enc = tok(ps, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        g = model.generate(**enc, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.pad_token_id)
        out += [x.strip() for x in tok.batch_decode(g[:, enc.input_ids.shape[1]:], skip_special_tokens=True)]
    return out
