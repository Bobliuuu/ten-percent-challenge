"""Tiny random Qwen3-architecture base (for CPU unit tests). python make_tiny.py OUT_DIR TOKENIZER_DIR(with tokenizer.json, tokenizer_config.json)"""
import json, os, shutil, sys, torch
from transformers import AutoConfig, AutoModelForCausalLM


def make(out, tokdir, layers=3, hidden=256, inter=512, vocab=151936, seed=0, tied=False):
    os.makedirs(out, exist_ok=True); torch.manual_seed(seed)
    cfg = json.load(open(os.path.join(tokdir, "config.json"))); cfg.update(hidden_size=hidden, intermediate_size=inter, num_hidden_layers=layers, num_attention_heads=4, num_key_value_heads=2, head_dim=64, max_window_layers=layers, vocab_size=vocab, tie_word_embeddings=tied)
    json.dump(cfg, open(os.path.join(out, "config.json"), "w")); conf = AutoConfig.from_pretrained(out)
    m = AutoModelForCausalLM.from_config(conf, dtype=torch.bfloat16)
    with torch.no_grad():
        for n, p in m.named_parameters():
            if p.ndim == 2: p.normal_(0, 0.05)
            elif "norm" in n: p.fill_(1.0)
    m.save_pretrained(out, safe_serialization=True)
    if tied:                                                           # like the real Qwen3-0.6B checkpoint: duplicate lm_head.weight stored in the file
        from safetensors.torch import load_file, save_file
        sd = load_file(os.path.join(out, "model.safetensors")); sd["lm_head.weight"] = sd["model.embed_tokens.weight"].clone(); save_file(sd, os.path.join(out, "model.safetensors"), metadata={"format": "pt"})
    for f in ("tokenizer.json", "tokenizer_config.json", "generation_config.json"): shutil.copy(os.path.join(tokdir, f), os.path.join(out, f))
    return out


if __name__ == "__main__": make(sys.argv[1], sys.argv[2])
