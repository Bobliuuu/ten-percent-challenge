"""REAL generation eval of INJ dev set: python vreal2.py MODEL_DIR OUT.json  (greedy, chat template, vLLM, max 16 new tokens, strict tournament scorer). Per item: output text + correct."""
import sys, json, os
md, outp = sys.argv[1:3]; R = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, R)
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import scorer
tok = AutoTokenizer.from_pretrained(md)
def chat(u):
    try: return tok.apply_chat_template([{"role": "user", "content": u}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError: return tok.apply_chat_template([{"role": "user", "content": u}], tokenize=False, add_generation_prompt=True)
items = [json.loads(l) for l in open(os.environ.get("INJ_DEV", R + "/dev_inj.jsonl"))]
llm = LLM(md, dtype="bfloat16", max_model_len=4096, gpu_memory_utilization=float(os.environ.get('VLLM_UTIL', '0.5')), enable_prefix_caching=False, tensor_parallel_size=int(os.environ.get('VLLM_TP', '1')))
ps = [tok.encode(chat(i["query"]), add_special_tokens=False)[-3800:] for i in items]
res = llm.generate([{"prompt_token_ids": p} for p in ps], SamplingParams(temperature=0, max_tokens=16, skip_special_tokens=False))
rows = [dict(id=i["id"], kind=i["kind"], source=i["source"], gold=i["gold"], pred=scorer.parse_label(r.outputs[0].text), correct=bool(scorer.correct(r.outputs[0].text, i["gold"])), out=r.outputs[0].text[:60]) for i, r in zip(items, res)]
json.dump(dict(model=md, rows=rows), open(outp, "w")); print("REAL_DONE", sum(r["correct"] for r in rows), len(rows), flush=True)
