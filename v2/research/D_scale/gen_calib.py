"""self-distilled calibration: base model greedy outputs on calib prompts. python gen_calib.py MODEL_DIR IN_calib.jsonl OUT_DIR N"""
import sys, json, os
md, inp, outd, n = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
tok = AutoTokenizer.from_pretrained(md); items = [json.loads(l) for l in open(inp)][:n]; ps = []
for it in items:
    tools = [{"type": "function", "function": t} for t in it["tools"]]
    try: p = tok.apply_chat_template([{"role": "user", "content": it["query"]}], tools=tools, add_generation_prompt=True, tokenize=False, enable_thinking=False)
    except TypeError: p = tok.apply_chat_template([{"role": "user", "content": it["query"]}], tools=tools, add_generation_prompt=True, tokenize=False)
    ps.append(tok.encode(p, add_special_tokens=False)[-12000:])
llm = LLM(md, dtype="bfloat16", max_model_len=14000, gpu_memory_utilization=0.92, enable_prefix_caching=False)
res = llm.generate([{"prompt_token_ids": p} for p in ps], SamplingParams(temperature=0, max_tokens=400, skip_special_tokens=False))
os.makedirs(outd, exist_ok=True)
with open(outd + "/calib_t1.jsonl", "w") as f:
    for it, r in zip(items, res):
        t = r.outputs[0].text
        if "<|im_end|>" not in t: t += "<|im_end|>"
        t = t.split("<|im_end|>")[0] + "<|im_end|>"
        it = dict(it); it["selfgen"] = t[:-len("<|im_end|>")]; f.write(json.dumps(it) + "\n")
for fn in ("probe_items.jsonl", "scorer.py"):
    import shutil; shutil.copy(os.path.dirname(inp) + "/" + fn, outd + "/" + fn)
print("GEN_DONE", len(items))
