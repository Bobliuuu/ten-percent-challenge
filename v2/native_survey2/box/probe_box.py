import os, json, sys
def run(model, tag, max_len=6144):
    D="/home/ubuntu/work/data"
    import random, time
    os.environ["HF_HOME"] = "/home/ubuntu/work/hf"
    from huggingface_hub import snapshot_download
    from vllm import LLM, SamplingParams
    src = model if os.path.isdir(model) else snapshot_download(model)
    t1 = [json.loads(l) for l in open(D+"/probe_items.jsonl")]
    TOXP = open(D+"/tox_prompt.txt").read(); INJP = open(D+"/inj_prompt.txt").read()
    cls = []
    for f in ("s1_dev.jsonl", "s3_dev.jsonl"): cls += [("tox", json.loads(l)) for l in open(D+"/tox_" + f)]
    for f in ("dev_S1.jsonl", "dev_S3.jsonl"): cls += [("inj", json.loads(l)) for l in open(D+"/inj_" + f)]
    random.Random(0).shuffle(cls); cls = cls[:1400]
    llm = LLM(src, dtype="bfloat16", max_model_len=max_len, gpu_memory_utilization=0.92, enforce_eager=False, limit_mm_per_prompt={"image": 0})
    tok = llm.get_tokenizer()
    def tpl(msgs, tools=None): return tok.apply_chat_template(msgs, tools=tools, add_generation_prompt=True, tokenize=False, enable_thinking=False)
    p1 = [tpl([{"role": "user", "content": it["query"]}], [{"type": "function", "function": t} for t in it["tools"]]) for it in t1]
    p2 = [tpl([{"role": "user", "content": (TOXP if k == "tox" else INJP).replace("{t}", r["text"])}]) for k, r in cls]
    t0 = time.time()
    o1 = llm.generate(p1, SamplingParams(temperature=0, max_tokens=300)); o2 = llm.generate(p2, SamplingParams(temperature=0, max_tokens=16))
    R = {"model": model, "t1": [{"id": it["id"], "kind": it["kind"], "source": it["source"], "gold": it["gold"], "out": o.outputs[0].text} for it, o in zip(t1, o1)],
         "cls": [{"task": k, "id": r["id"], "gold": r["gold"], "kind": r.get("kind"), "source": r.get("source"), "out": o.outputs[0].text} for (k, r), o in zip(cls, o2)], "secs": time.time() - t0}
    os.makedirs("/home/ubuntu/work/out", exist_ok=True); json.dump(R, open(f"/home/ubuntu/work/out/{tag}.json", "w")); print("DONE", tag, R["secs"])

if __name__=='__main__': run(sys.argv[1], sys.argv[2], *(int(x) for x in sys.argv[3:]))
