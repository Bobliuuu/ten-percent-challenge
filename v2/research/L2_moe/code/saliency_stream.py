"""Layer-streaming saliency for models too big for GPU memory. usage: saliency_stream.py SNAPDIR OUT.pt n_tool n_gen
Per layer: load that layer's weights to GPU, run all calibration sequences (right-padded, causal), record freq/REAP/EAN stats, advance hidden states. One pass over the weights."""
import sys, json, random, torch, time
sys.path.insert(0, "/opt/dlami/nvme/work/l2/code")
from moe_lib import *
import torch.nn as nn
from transformers import AutoTokenizer, AutoConfig
from pruned_loader import load_pruned
import transformers.models.qwen3_moe.modeling_qwen3_moe as M
from safetensors import safe_open
snap, out, n_tool, n_gen = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
cfg = AutoConfig.from_pretrained(snap); cfg._attn_implementation = "sdpa"; tok = AutoTokenizer.from_pretrained(snap)
items = [json.loads(l) for l in open("/opt/dlami/nvme/work/l2/data/s1_calib.jsonl")]
random.Random(0).shuffle(items); items = items[:n_tool]
texts = [chat_prompt(tok, it) + gold_qwen(it) + "<|im_end|>" for it in items]
if n_gen: texts += [json.loads(l)["text"] for l in open("/opt/dlami/nvme/work/l2/data/general.jsonl")][:n_gen]
texts.sort(key=len); tok.padding_side = "right"
dev = torch.device("cuda:0")
wm = json.load(open(f"{snap}/model.safetensors.index.json"))["weight_map"]; files = {}
def get(n):
    f = wm[n]
    if f not in files: files[f] = safe_open(f"{snap}/{f}", "pt", device="cpu")
    return files[f].get_tensor(n)
emb = get("model.embed_tokens.weight").to(dev)
batches = []
for i in range(0, len(texts), 8):
    enc = tok(texts[i:i+8], return_tensors="pt", padding=True, truncation=True, max_length=2048)
    ids = enc.input_ids.to(dev); valid = enc.attention_mask.bool().to(dev)
    batches.append((torch.nn.functional.embedding(ids, emb).bfloat16(), valid))
del emb
L, E = cfg.num_hidden_layers, cfg.num_experts
st = Stats(L, E, dev); STATE["stats"] = st
rot = M.Qwen3MoeRotaryEmbedding(cfg, device=dev)
M.Qwen3MoeSparseMoeBlock.forward = fwd
t0 = time.time()
for l in range(L):
    with torch.device("meta"): lay = M.Qwen3MoeDecoderLayer(cfg, l)
    lay.mlp._li = l
    pre = f"model.layers.{l}."
    sd = {k[len(pre):]: get(k).to(dev, torch.bfloat16) for k in wm if k.startswith(pre)}
    lay = lay.to_empty(device=dev); lay.load_state_dict(sd, strict=True); del sd
    lay.eval()
    with torch.no_grad():
        for bi, (h, valid) in enumerate(batches):
            STATE["valid"] = valid
            pos = torch.arange(h.shape[1], device=dev)[None].expand(h.shape[0], -1)
            pe = rot(h, pos)
            o = lay(h, attention_mask=None, position_ids=pos, position_embeddings=pe)
            batches[bi] = ((o[0] if isinstance(o, tuple) else o), valid)
    del lay; torch.cuda.empty_cache()
    print("layer", l, round(time.time() - t0), flush=True)
    if l % 10 == 9 or l == L - 1:
        c = st.cnt.clamp(min=1)
        torch.save(dict(cnt=st.cnt.cpu(), reap=(st.reap / c).cpu(), ean=(st.ean / c).cpu(), gatesum=st.gate.cpu(), routprob=(st.rl_mean / st.ntok[:, None].clamp(min=1)).cpu(), upto=l, n_tool=n_tool, n_gen=n_gen), out)
print("done")
