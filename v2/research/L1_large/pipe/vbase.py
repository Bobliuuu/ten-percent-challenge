"""base reference via vLLM (fp8 weights if --fp8; 70B bf16 does not fit one 96GB GPU). usage: vbase.py MODEL_DIR OUT.json [fp8]"""
import sys, json, time
sys.path.insert(0, "/opt/dlami/nvme/work/L1/t1v2")
from scorer import score_item
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
md, out = sys.argv[1], sys.argv[2]; fp8 = len(sys.argv) > 3
items = [json.loads(l) for l in open("/opt/dlami/nvme/work/L1/t1v2/probe_items.jsonl")]
tok = AutoTokenizer.from_pretrained(md); prompts = []
for it in items:
    tools = [{"type": "function", "function": t} for t in it["tools"]]
    try: p = tok.apply_chat_template([{"role": "user", "content": it["query"]}], tools=tools, add_generation_prompt=True, tokenize=False, enable_thinking=False)
    except TypeError: p = tok.apply_chat_template([{"role": "user", "content": it["query"]}], tools=tools, add_generation_prompt=True, tokenize=False)
    prompts.append(p)
t0 = time.time()
kw = dict(quantization="fp8") if fp8 else {}
llm = LLM(md, max_model_len=16000, gpu_memory_utilization=0.92, seed=0, enable_prefix_caching=True, **kw)
outs = llm.generate(prompts, SamplingParams(temperature=0, max_tokens=400, skip_special_tokens=False), use_tqdm=False)
rows = []
for it, o in zip(items, outs):
    t = o.outputs[0].text; s = score_item(t, it)
    rows.append(dict(id=it["id"], source=it["source"], kind=it["kind"], correct=bool(s["correct"]), parse_fail=bool(s["parse_fail"]), output=t[:600]))
json.dump(dict(model=md, fp8=fp8, t1=rows, secs=time.time() - t0), open(out, "w"))
import collections
c = collections.defaultdict(list)
for r in rows: c[(r["source"], r["kind"])].append(r["correct"])
print("BASE_DONE", sum(r["correct"] for r in rows), len(rows), {k: round(sum(v) / len(v), 3) for k, v in c.items()})
