"""probe2_box.py MODEL TAG ITEMS.jsonl : vLLM greedy over a generic item list (id, user, tools?, maxnew). -> ~/work/out2/TAG.json {id: text}"""
import os, json, sys, time
os.environ["HF_HOME"] = "/home/ubuntu/work/hf"
from huggingface_hub import snapshot_download
from vllm import LLM, SamplingParams
def main():
    model, tag, items_p = sys.argv[1], sys.argv[2], sys.argv[3]
    src = model if os.path.isdir(model) else snapshot_download(model)
    items = [json.loads(l) for l in open(items_p)]
    llm = LLM(src, dtype="bfloat16", max_model_len=6144, gpu_memory_utilization=0.92, max_num_seqs=256, limit_mm_per_prompt={"image": 0})
    tok = llm.get_tokenizer()
    def tpl(it):
        tools = [{"type": "function", "function": t} for t in it["tools"]] if it.get("tools") else None
        return tok.apply_chat_template([{"role": "user", "content": it["user"]}], tools=tools, add_generation_prompt=True, tokenize=False, enable_thinking=False)
    res = {}; t0 = time.time()
    for mn in sorted(set(i["maxnew"] for i in items)):
        sub = [i for i in items if i["maxnew"] == mn]
        outs = llm.generate([tpl(i) for i in sub], SamplingParams(temperature=0, max_tokens=mn))
        for i, o in zip(sub, outs): res[i["id"]] = o.outputs[0].text
    os.makedirs("/home/ubuntu/work/out2", exist_ok=True); json.dump(res, open(f"/home/ubuntu/work/out2/{tag}.json", "w")); print("DONE", tag, time.time() - t0)

if __name__ == '__main__':
    main()
