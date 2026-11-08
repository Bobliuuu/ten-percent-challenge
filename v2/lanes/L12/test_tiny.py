"""local smoke test of l12_core on a tiny random Qwen3Moe (CPU, fp32)"""
import os, json, sys, torch, tempfile, shutil
from transformers import AutoConfig, AutoModelForCausalLM
from huggingface_hub import snapshot_download
from safetensors.torch import save_file
d = "/private/tmp/claude-501/-Users-jhome/4f4f15d2-8745-4370-90f5-19fb51a5941b/scratchpad/tiny"; shutil.rmtree(d, ignore_errors=True); shutil.rmtree(d + "_out", ignore_errors=True); os.makedirs(d)
tk = snapshot_download("Qwen/Qwen3-30B-A3B-Instruct-2507", allow_patterns=["tokenizer*", "vocab.json", "merges.txt", "generation_config.json"]); 
for f in os.listdir(tk): shutil.copy(os.path.join(tk, f), d)
cfg = AutoConfig.from_pretrained("Qwen/Qwen3-30B-A3B-Instruct-2507"); cfg.hidden_size = 64; cfg.num_hidden_layers = 3; cfg.num_experts = 16; cfg.num_experts_per_tok = 4; cfg.moe_intermediate_size = 128
cfg.intermediate_size = 128; cfg.num_attention_heads = 4; cfg.num_key_value_heads = 2; cfg.head_dim = 16; cfg.decoder_sparse_step = 1; cfg.mlp_only_layers = []
cfg.save_pretrained(d)
torch.manual_seed(0); m = AutoModelForCausalLM.from_config(cfg, dtype=torch.float32)
sd = {}
for k, v in m.state_dict().items():
    if "experts.gate_up_proj" in k:
        l = k.split(".")[2]; I = v.shape[1] // 2
        for e in range(v.shape[0]): sd[f"model.layers.{l}.mlp.experts.{e}.gate_proj.weight"] = v[e, :I].contiguous(); sd[f"model.layers.{l}.mlp.experts.{e}.up_proj.weight"] = v[e, I:].contiguous()
    elif "experts.down_proj" in k:
        l = k.split(".")[2]
        for e in range(v.shape[0]): sd[f"model.layers.{l}.mlp.experts.{e}.down_proj.weight"] = v[e].contiguous()
    else: sd[k] = v.contiguous()
print([k for k in sd if "layers.0." in k and "experts" not in k])
save_file(sd, d + "/model-00001-of-00001.safetensors"); tot = sum(v.numel() * 2 for v in sd.values())
json.dump(dict(metadata=dict(total_size=tot), weight_map={k: "model-00001-of-00001.safetensors" for k in sd}), open(d + "/model.safetensors.index.json", "w"))
# ref logits
ids = torch.tensor([[1, 5, 9, 20, 33, 7]]); ref = m(ids).logits
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import l12_core as C
eng = C.Engine("tiny", "data", d + "_out", dtype=torch.float32, dev="cpu", snap=d)
eng.set_full_all(); eng.restore_nonexpert()
got = eng.lm(ids).logits; print("max logit diff vs HF", (got - ref).abs().max().item())
eng.D["cal"] = eng.D["cal"][:60]; eng.D["sel"] = eng.D["sel"][:30]; eng.D["rep"] = eng.D["rep"][:20]
eng.D["sel"] = eng.D["sel"][:30]
_o = eng.gen_eval(eng.D["sel"]); 
for r, o in zip(eng.D["sel"], _o): r["answer"] = o
for r in eng.D["rep"]: r["answer"] = "x"
eng.run_cfg(dict(name="base"))
eng.run_stats(60)
for cfg in [dict(name="t_prune", score="reap", keep=0.5, eb=3, eg=16, attn_g=16, gchunk=8, attn_b=4),
            dict(name="t_merge", score="mass", keep=0.5, merge="align", eb=3, eg=16),
            dict(name="t_sub", score="mass", keep=0.5, subspace=dict(r=48, cb=4, r_down=24), eg=16),
            dict(name="t_gptq", score="mass", keep=0.5, eb=3, eg=16, attn_g=16, gchunk=8, quant="gptq", nh=40, nmin=4),
            dict(name="t_k4", score="reap", keep=0.5, topk=2, eb=4, eg=16, adapt="global", kmin=2),
            dict(name="t_kl", score="kl", keep=0.5, adapt="kl", eb=4, eg=16)]:
    eng.run_cfg(dict(cfg, sets=["sel"]))
