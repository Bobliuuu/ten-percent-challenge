"""NEWS real-generation eval: python vreal_news.py MODEL OUT.json SET.jsonl   (greedy, chat template, max_new_tokens 8, strict parse_news from v2/exp/run_retention.py)"""
import sys, json, os, importlib.util
md, outp, setf = sys.argv[1:4]
spec = importlib.util.spec_from_file_location("rr", os.environ.get("RR", "/opt/dlami/nvme/e32/run_retention.py")); rr = importlib.util.module_from_spec(spec); spec.loader.exec_module(rr)
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
tok = AutoTokenizer.from_pretrained(md)
chat = lambda u: tok.apply_chat_template([{"role": "user", "content": u}], tokenize=False, add_generation_prompt=True)
items = [json.loads(l) for l in open(setf)]
llm = LLM(md, dtype="bfloat16", max_model_len=4096, gpu_memory_utilization=float(os.environ.get('VLLM_UTIL', '0.5')), enable_prefix_caching=False)
ps = [tok.encode(chat(i["user"]), add_special_tokens=False)[-3800:] for i in items]
res = llm.generate([{"prompt_token_ids": p} for p in ps], SamplingParams(temperature=0, max_tokens=8, skip_special_tokens=False))
rows = [dict(id=i["id"], kind=i["kind"], source=i["src"], gold=i["answer"].lower(), pred=rr.parse_news(r.outputs[0].text), correct=bool(rr.news_correct(r.outputs[0].text, i["answer"])), out=r.outputs[0].text[:40]) for i, r in zip(items, res)]
json.dump(dict(model=md, rows=rows), open(outp, "w")); print("REAL_DONE", sum(r["correct"] for r in rows), len(rows), flush=True)
