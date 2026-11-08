"""REAL greedy generation on INJ dev (4313, strict tournament scorer) and NEWS dev (3500, exact-string label) with one vLLM load.
usage: python vreal_both.py MODEL_DIR OUT.json"""
import sys, json, os
md, outp = sys.argv[1:3]; R = "/code"; sys.path.insert(0, R)
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import scorer
tok = AutoTokenizer.from_pretrained(md)
def chat(u):
    try: return tok.apply_chat_template([{"role": "user", "content": u}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError: return tok.apply_chat_template([{"role": "user", "content": u}], tokenize=False, add_generation_prompt=True)
inj = [json.loads(l) for l in open(R + "/dev_inj.jsonl")]
news = [json.loads(l) for l in open(R + "/items_dev_ext.jsonl")]
llm = LLM(md, dtype="bfloat16", max_model_len=4096, gpu_memory_utilization=float(os.environ.get('VLLM_UTIL', '0.5')), enable_prefix_caching=False, tensor_parallel_size=int(os.environ.get('VLLM_TP', '1')))
sp = SamplingParams(temperature=0, max_tokens=16, skip_special_tokens=False)
enc = lambda t: tok.encode(chat(t), add_special_tokens=False)[-3800:]
r1 = llm.generate([{"prompt_token_ids": enc(i["query"])} for i in inj], sp)
rows_inj = [dict(id=i["id"], kind=i["kind"], source=i["source"], gold=i["gold"], correct=bool(scorer.correct(r.outputs[0].text, i["gold"])), out=r.outputs[0].text[:60]) for i, r in zip(inj, r1)]
r2 = llm.generate([{"prompt_token_ids": enc(i["user"])} for i in news], sp)
def nc(t, g): return t.replace("<|im_end|>", "").strip() == g
rows_news = [dict(id=i["id"], kind=i["kind"], source=i["role"], gold=i["answer"], correct=bool(nc(r.outputs[0].text, i["answer"])), out=r.outputs[0].text[:60]) for i, r in zip(news, r2)]
json.dump(dict(model=md, inj=rows_inj, news=rows_news), open(outp, "w")); print("REAL_DONE inj", sum(r["correct"] for r in rows_inj), len(rows_inj), "news", sum(r["correct"] for r in rows_news), len(rows_news), flush=True)
