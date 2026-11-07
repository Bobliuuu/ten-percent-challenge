"""python gen_ct_self.py MODEL_DIR IN.jsonl OUT.jsonl : replace 'answer' by the base model's own greedy output (max 32 tok) for each calibration item"""
import sys, json
md, inp, outp = sys.argv[1:4]
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
tok = AutoTokenizer.from_pretrained(md); items = [json.loads(l) for l in open(inp)]
def chat(u):
    try: return tok.apply_chat_template([{"role": "user", "content": u}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError: return tok.apply_chat_template([{"role": "user", "content": u}], tokenize=False, add_generation_prompt=True)
llm = LLM(md, dtype="bfloat16", max_model_len=8192, gpu_memory_utilization=0.92, enable_prefix_caching=False)
res = llm.generate([{"prompt_token_ids": tok.encode(chat(i["user"]), add_special_tokens=False)[-7500:]} for i in items], SamplingParams(temperature=0, max_tokens=32, skip_special_tokens=True))
n = 0
with open(outp, "w") as f:
    for i, r in zip(items, res):
        t = r.outputs[0].text.strip(); n += t == i["answer"]; i = dict(i); i["gold_answer"] = i["answer"]; i["answer"] = t; f.write(json.dumps(i) + "\n")
print("GEN_CT_DONE agree-with-gold", n, len(items))
