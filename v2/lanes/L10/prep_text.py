"""Extract the text-only self-attention path of Llama-3.2-11B-Vision (mllama) into a plain LlamaForCausalLM checkpoint.
Drops: vision tower, projector, 8 cross-attention layers (inactive when no image). Moves only tensors, no training."""
import json, os, glob, re, torch
def ensure(src, out, verify=False, kind="mllama"):
    if kind == "omni": return ensure_omni(src, out)
    if os.path.exists(out + "/DONE"): return out
    from huggingface_hub import snapshot_download
    from safetensors import safe_open
    from safetensors.torch import save_file
    d = snapshot_download(src, allow_patterns=["*.safetensors", "*.json", "tokenizer*", "*.model"])
    cfg = json.load(open(d + "/config.json")); tc = cfg["text_config"]; xl = list(tc["cross_attention_layers"]); n_all = tc["num_hidden_layers"]
    keep = [i for i in range(n_all) if i not in xl]; remap = {o: n for n, o in enumerate(keep)}
    sd = {}
    for f in sorted(glob.glob(d + "/*.safetensors")):
        with safe_open(f, "pt") as g:
            for k in g.keys():
                if not k.startswith("language_model."): continue
                k2 = k[len("language_model."):]
                m = re.match(r"model\.layers\.(\d+)\.(.*)", k2)
                if m:
                    i = int(m.group(1))
                    if i in xl: continue
                    k2 = f"model.layers.{remap[i]}.{m.group(2)}"
                sd[k2] = g.get_tensor(k).to(torch.bfloat16)
    os.makedirs(out, exist_ok=True)
    lc = dict(architectures=["LlamaForCausalLM"], model_type="llama", hidden_size=tc["hidden_size"], intermediate_size=tc["intermediate_size"], num_attention_heads=tc["num_attention_heads"],
              num_key_value_heads=tc["num_key_value_heads"], num_hidden_layers=len(keep), vocab_size=sd["model.embed_tokens.weight"].shape[0], max_position_embeddings=tc["max_position_embeddings"],
              rms_norm_eps=tc["rms_norm_eps"], rope_theta=tc["rope_theta"], rope_scaling=tc.get("rope_scaling"), tie_word_embeddings=False, torch_dtype="bfloat16", hidden_act="silu",
              bos_token_id=tc.get("bos_token_id", 128000), eos_token_id=tc.get("eos_token_id", [128001, 128008, 128009]), pad_token_id=tc.get("pad_token_id", 128004))
    json.dump(lc, open(out + "/config.json", "w"), indent=1)
    save_file(sd, out + "/model.safetensors", metadata={"format": "pt"})
    for f in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "generation_config.json", "chat_template.json"):
        if os.path.exists(d + "/" + f): os.system(f"cp {d}/{f} {out}/{f}")
    if verify:   # base identity: text-only greedy of mllama original vs extracted
        from transformers import AutoTokenizer, AutoModelForCausalLM, MllamaForConditionalGeneration
        tok = AutoTokenizer.from_pretrained(d); P = ["Which skill handles: set an alarm for 7am? Options Alarm, Weather, Music.", "What is the capital of France?", "Write one sentence about cats."]
        ids = [tok.apply_chat_template([{"role": "user", "content": [{"type": "text", "text": p}]}], add_generation_prompt=True, tokenize=False) for p in P]
        a = MllamaForConditionalGeneration.from_pretrained(d, torch_dtype=torch.bfloat16).cuda().eval()
        b = AutoModelForCausalLM.from_pretrained(out, torch_dtype=torch.bfloat16).cuda().eval(); same = []
        for t in ids:
            x = tok(t, return_tensors="pt", add_special_tokens=False).to("cuda")
            ga = a.generate(**x, max_new_tokens=24, do_sample=False); gb = b.generate(**x, max_new_tokens=24, do_sample=False)
            same.append(bool((ga[0] == gb[0]).all())); print("VERIFY", same[-1], tok.decode(gb[0][x["input_ids"].shape[1]:]), flush=True)
        json.dump(dict(same=same, dropped_layers=xl), open(out + "/verify.json", "w")); del a, b
    open(out + "/DONE", "w").write("1"); return out


def ensure_omni(src, out):
    """Qwen2.5-Omni thinker text path -> plain Qwen2ForCausalLM (drops visual, audio_tower, talker, token2wav). Tensor moves only."""
    if os.path.exists(out + "/DONE"): return out
    from huggingface_hub import snapshot_download
    from safetensors import safe_open
    from safetensors.torch import save_file
    d = snapshot_download(src, allow_patterns=["*.safetensors", "*.json", "tokenizer*", "*.model", "merges.txt", "vocab.json"])
    cfg = json.load(open(d + "/config.json")); tc = cfg["thinker_config"]["text_config"]; sd = {}
    for f in sorted(glob.glob(d + "/*.safetensors")):
        with safe_open(f, "pt") as g:
            for k in g.keys():
                if k.startswith("thinker.model.") or k.startswith("thinker.lm_head."): sd[k[len("thinker."):]] = g.get_tensor(k).to(torch.bfloat16)
    os.makedirs(out, exist_ok=True)
    lc = dict(architectures=["Qwen2ForCausalLM"], model_type="qwen2", hidden_size=tc["hidden_size"], intermediate_size=tc["intermediate_size"], num_attention_heads=tc["num_attention_heads"],
              num_key_value_heads=tc["num_key_value_heads"], num_hidden_layers=tc["num_hidden_layers"], vocab_size=sd["model.embed_tokens.weight"].shape[0], max_position_embeddings=tc["max_position_embeddings"],
              rms_norm_eps=tc["rms_norm_eps"], rope_theta=tc.get("rope_theta", 1000000.0), tie_word_embeddings=False, torch_dtype="bfloat16", hidden_act="silu", use_sliding_window=False,
              bos_token_id=151643, eos_token_id=[151645, 151643], pad_token_id=151643)
    json.dump(lc, open(out + "/config.json", "w"), indent=1)
    save_file(sd, out + "/model.safetensors", metadata={"format": "pt"})
    for f in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "generation_config.json", "chat_template.json", "vocab.json", "merges.txt"):
        if os.path.exists(d + "/" + f): os.system(f"cp {d}/{f} {out}/{f}")
    # chat template may live in chat_template.json
    if os.path.exists(out + "/chat_template.json"):
        ct = json.load(open(out + "/chat_template.json")); tk = json.load(open(out + "/tokenizer_config.json")); tk["chat_template"] = ct.get("chat_template", ct); json.dump(tk, open(out + "/tokenizer_config.json", "w"))
    open(out + "/DONE", "w").write("1"); return out
