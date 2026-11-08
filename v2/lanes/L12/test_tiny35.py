"""local smoke test of the Qwen3.5-MoE path (tiny random, CPU fp32)"""
import os, json, sys, torch, shutil
from transformers import AutoConfig, AutoModelForCausalLM
from huggingface_hub import snapshot_download
from safetensors.torch import save_file
d = "/private/tmp/claude-501/-Users-jhome/4f4f15d2-8745-4370-90f5-19fb51a5941b/scratchpad/tiny35"; shutil.rmtree(d, ignore_errors=True); shutil.rmtree(d + "_out", ignore_errors=True); os.makedirs(d)
tk = snapshot_download("Qwen/Qwen3.5-35B-A3B", allow_patterns=["tokenizer*", "vocab.json", "merges.txt", "generation_config.json", "chat_template*"])
for f in os.listdir(tk): shutil.copy(os.path.join(tk, f), d)
cfg = AutoConfig.from_pretrained("Qwen/Qwen3.5-35B-A3B"); tc = cfg.text_config
tc.hidden_size = 64; tc.num_hidden_layers = 4; tc.layer_types = tc.layer_types[:4]; tc.num_experts = 16; tc.num_experts_per_tok = 4; tc.moe_intermediate_size = 32; tc.shared_expert_intermediate_size = 32
tc.num_attention_heads = 4; tc.num_key_value_heads = 2; tc.head_dim = 16; tc.linear_key_head_dim = 16; tc.linear_value_head_dim = 16; tc.linear_num_key_heads = 2; tc.linear_num_value_heads = 4
cfg.save_pretrained(d)
torch.manual_seed(0); m = AutoModelForCausalLM.from_config(tc, dtype=torch.float32)
sd = {}
for k, v in m.state_dict().items(): sd[("model.language_model." + k[6:]) if k.startswith("model.") else k] = v.contiguous()
print(list(sd)[:6], [k for k in sd if "layers.0." in k and "experts" not in k][:14])
save_file(sd, d + "/model-00001-of-00001.safetensors"); tot = sum(v.numel() * 2 for v in sd.values())
json.dump(dict(metadata=dict(total_size=tot), weight_map={k: "model-00001-of-00001.safetensors" for k in sd}), open(d + "/model.safetensors.index.json", "w"))
ids = torch.tensor([[1, 5, 9, 20, 33, 7]]); ref = m(ids).logits
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import l12_core as C
eng = C.Engine("tiny35", "data", d + "_out", dtype=torch.float32, dev="cpu", snap=d)
eng.set_full_all(); eng.restore_nonexpert(); got = eng.lm(ids).logits; print("max logit diff vs HF", (got - ref).abs().max().item())
eng.D["cal"] = eng.D["cal"][:40]; eng.D["sel"] = eng.D["sel"][:20]; eng.D["rep"] = eng.D["rep"][:10]
for r in eng.D["sel"]: r["answer"] = "x"
for r in eng.D["rep"]: r["answer"] = "x"
eng.run_cfg(dict(name="base")); eng.run_stats(40)
for cfg in [dict(name="q_prune", score="reap", keep=0.5, eb=3, eg=16, attn_g=16, attn_b=4, shared_b=4),
            dict(name="q_gptq", score="mass", keep=0.5, eb=3, eg=16, quant="gptq", nh=30, nmin=4, gchunk=8),
            dict(name="q_merge", score="mass", keep=0.5, merge="align", eb=3, eg=16)]:
    eng.run_cfg(dict(cfg, sets=["sel"]))
