"""Compute per-layer expert saliency (freq, REAP, EAN) on calibration text. usage: saliency.py MODEL OUT.pt [n_tool] [n_general]
Calibration = s1_calib items (tool-call, multi-heavy: prompt in native template + gold call in native format), + general slice (optional). NEVER locked/S2."""
import sys, json, random, torch
sys.path.insert(0, "/opt/dlami/nvme/work/l2/code")
from moe_lib import *
from transformers import AutoTokenizer, AutoModelForCausalLM
model_id, out = sys.argv[1], sys.argv[2]; n_tool = int(sys.argv[3]) if len(sys.argv) > 3 else 800; n_gen = int(sys.argv[4]) if len(sys.argv) > 4 else 0
tok = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto"); model.eval()
L = patch(model); E = model.config.num_experts
items = [json.loads(l) for l in open("/opt/dlami/nvme/work/l2/data/s1_calib.jsonl")]
random.Random(0).shuffle(items)
# stratify: keep multi-heavy as in pool ratio; take first n_tool
items = items[:n_tool]
texts = [chat_prompt(tok, it) + gold_qwen(it) + "<|im_end|>" for it in items]
if n_gen:
    gen = [json.loads(l)["text"] for l in open("/opt/dlami/nvme/work/l2/data/general.jsonl")][:n_gen]; texts += gen
STATE["stats"] = st = Stats(L, E, model.device)
ntok = 0
with torch.no_grad():
    tok.padding_side = "right"
    texts.sort(key=len)
    for i in range(0, len(texts), 16):
        enc = tok(texts[i:i+16], return_tensors="pt", padding=True, truncation=True, max_length=3072).to(model.device)
        STATE["valid"] = enc.attention_mask.bool()
        model(**enc, use_cache=False); ntok += int(enc.attention_mask.sum())
        if i % 160 == 0: print(i, len(texts), flush=True)
c = st.cnt.clamp(min=1)
torch.save(dict(cnt=st.cnt.cpu(), reap=(st.reap / c).cpu(), ean=(st.ean / c).cpu(), gatesum=st.gate.cpu(), routprob=(st.rl_mean / st.ntok[:, None]).cpu(), ntok=ntok, n_tool=n_tool, n_gen=n_gen), out)
print("done", ntok)
