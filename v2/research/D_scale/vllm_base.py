"""vLLM T1 base eval: python vllm_base.py MODEL_DIR OUT.json T1DIR [fp8|bf16] [sub]"""
import sys, json, time
md, out, t1dir, prec = sys.argv[1:5]; sub = int(sys.argv[5]) if len(sys.argv) > 5 else 1
sys.path.insert(0, t1dir); from scorer import score_item
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
tok = AutoTokenizer.from_pretrained(md)
items = [json.loads(l) for l in open(t1dir + "/probe_items.jsonl")][::sub]; prompts = []
for it in items:
    tools = [{"type": "function", "function": t} for t in it["tools"]]
    try: p = tok.apply_chat_template([{"role": "user", "content": it["query"]}], tools=tools, add_generation_prompt=True, tokenize=False, enable_thinking=False)
    except TypeError: p = tok.apply_chat_template([{"role": "user", "content": it["query"]}], tools=tools, add_generation_prompt=True, tokenize=False)
    prompts.append(tok.encode(p, add_special_tokens=False)[-12000:])
import os
llm = LLM(md, tensor_parallel_size=int(os.environ.get("TP","1")), quantization=("fp8" if prec == "fp8" else None), dtype="bfloat16", max_model_len=14000, gpu_memory_utilization=0.92, enable_prefix_caching=False, max_num_seqs=int(os.environ.get('MAXSEQS','256')))
t0 = time.time()
res = llm.generate([{"prompt_token_ids": p} for p in prompts], SamplingParams(temperature=0, max_tokens=400, skip_special_tokens=False))
rows = []
for it, r in zip(items, res):
    t = r.outputs[0].text; s_ = score_item(t, it)
    rows.append(dict(id=it["id"], source=it["source"], kind=it["kind"], correct=bool(s_["correct"]), parse_fail=bool(s_["parse_fail"]), output=t[:600]))
json.dump(dict(model=md, prec=prec, t1=rows, secs=time.time() - t0), open(out, "w")); print("VLLM_DONE", sum(r["correct"] for r in rows), len(rows), time.time() - t0)
