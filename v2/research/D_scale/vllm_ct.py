"""Label-task retention via vLLM: python vllm_ct.py MODEL_DIR OUT.json ITEMS.jsonl [tasks_csv]. correct = every answer token (incl. <|im_end|>) is the argmax at its position (== greedy decoding reproduces the label), same criterion as C_tasks/dense_c.py run_ct."""
import sys, json, os, time
md, out, items_f = sys.argv[1:4]; tasks = sys.argv[4].split(",") if len(sys.argv) > 4 else None
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
tok = AutoTokenizer.from_pretrained(md)
items = [json.loads(l) for l in open(items_f)]
if tasks: items = [i for i in items if i["task"] in tasks]
def chat(p):
    try: return tok.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True, tokenize=False, enable_thinking=False)
    except TypeError: return tok.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True, tokenize=False)
eosid = tok.convert_tokens_to_ids("<|im_end|>"); enc = []
for it in items:
    p = tok.encode(chat(it["user"]), add_special_tokens=False); a = tok.encode(it["answer"], add_special_tokens=False) + [eosid]; enc.append((p, a))
llm = LLM(md, tensor_parallel_size=int(os.environ.get("TP", "1")), dtype="bfloat16", max_model_len=8192, gpu_memory_utilization=0.92, enable_prefix_caching=False)
t0 = time.time()
res = llm.generate([{"prompt_token_ids": (p + a)[-8000:]} for p, a in enc], SamplingParams(temperature=0, max_tokens=1, prompt_logprobs=1))
rows = []
for it, (p, a), r in zip(items, enc, res):
    pl = r.prompt_logprobs; full = (p + a)[-8000:]; n = len(full); ok = True
    for k in range(len(a)):
        pos = n - len(a) + k; d = pl[pos]; tid = full[pos]
        top = [t for t, lp in d.items() if lp.rank == 1]
        if tid not in top: ok = False; break
    rows.append(dict(id=it["id"], task=it["task"], role=it["role"], src=it["src"], kind=it["kind"], correct=ok))
json.dump(dict(model=md, rows=rows, secs=time.time() - t0), open(out, "w")); print("CT_DONE", sum(r["correct"] for r in rows), len(rows), time.time() - t0)
