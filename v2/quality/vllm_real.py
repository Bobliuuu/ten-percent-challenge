"""REAL-exam generation eval: python vllm_real.py MODEL_DIR OUTPREFIX inj,toolsel   (items/scorers in ./real). Greedy, chat template, vLLM; strict task scorers. writes OUTPREFIX_<task>.json"""
import sys, json, os, time
md, outp, tasks = sys.argv[1:4]; R = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, R)
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import inj_scorer, toolsel_scorer
tok = AutoTokenizer.from_pretrained(md)
def chat(u):
    try: return tok.apply_chat_template([{"role": "user", "content": u}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError: return tok.apply_chat_template([{"role": "user", "content": u}], tokenize=False, add_generation_prompt=True)
llm = LLM(md, dtype="bfloat16", max_model_len=8192, gpu_memory_utilization=float(os.environ.get('VLLM_UTIL','0.5')), enable_prefix_caching=False, tensor_parallel_size=int(os.environ.get('VLLM_TP','1')))
for task in tasks.split(","):
    if task == "inj":
        items = [json.loads(l) for l in open(R + "/inj_items.jsonl")]; U = [i["query"] for i in items]; mt = 16
        sc = lambda t, i: inj_scorer.parse_label(t) == i["gold_label"]; cell = lambda i: (i["source"], i["kind"])
    else:
        items = [json.loads(l) for l in open(R + "/toolsel_items.jsonl")]; U = [i["user"] for i in items]; mt = 32
        sc = lambda t, i: toolsel_scorer.score_toolsel(t, i)["correct"]; cell = lambda i: (i["role"], i["kind"])
    ps = [tok.encode(chat(u), add_special_tokens=False)[-7500:] for u in U]
    res = llm.generate([{"prompt_token_ids": p} for p in ps], SamplingParams(temperature=0, max_tokens=mt, skip_special_tokens=False))
    rows = [dict(id=i["id"], cell=list(cell(i)), correct=bool(sc(r.outputs[0].text, i)), out=r.outputs[0].text[:80]) for i, r in zip(items, res)]
    json.dump(dict(model=md, task=task, rows=rows), open(f"{outp}_{task}.json", "w")); print("REAL_DONE", task, sum(r["correct"] for r in rows), len(rows), flush=True)
