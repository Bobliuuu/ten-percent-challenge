"""Retention probe for 27B-class models (dev files only). modal run --detach probe27.py::main
Each model: vLLM greedy, T1 probe (600, 300 tok) + 1400 TOX/INJ dev (16 tok). Raw outputs -> volume tenpct-native2:/out/<tag>.json"""
import modal, os, json
H = os.path.expanduser("~/ten-percent/v2"); N = H + "/native_lowbit"
app = modal.App("tenpct-native2-probe")
img = (modal.Image.debian_slim(python_version="3.11").pip_install("vllm", "huggingface_hub", "hf_transfer")
       .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
       .add_local_file(H + "/tournament/T1/probe_items.jsonl", "/app/probe_items.jsonl")
       .add_local_file(N + "/tox_prompt.txt", "/app/tox_prompt.txt").add_local_file(N + "/inj_prompt.txt", "/app/inj_prompt.txt"))
vol = modal.Volume.from_name("tenpct-native2", create_if_missing=True)
toxv = modal.Volume.from_name("tenpct-tox-exam"); injv = modal.Volume.from_name("tenpct-inj-exam")
@app.function(image=img, gpu="H100", cpu=8, memory=98304, timeout=5*3600, volumes={"/vol": vol, "/tox": toxv, "/inj": injv})
def run(model, tag, max_len=6144):
    import random, time
    os.environ["HF_HOME"] = "/vol/hf"
    from huggingface_hub import snapshot_download
    from vllm import LLM, SamplingParams
    src = snapshot_download(model, allow_patch=None) if False else snapshot_download(model)
    t1 = [json.loads(l) for l in open("/app/probe_items.jsonl")]
    TOXP = open("/app/tox_prompt.txt").read(); INJP = open("/app/inj_prompt.txt").read()
    cls = []
    for f in ("s1_dev.jsonl", "s3_dev.jsonl"): cls += [("tox", json.loads(l)) for l in open("/tox/data/" + f)]
    for f in ("dev_S1.jsonl", "dev_S3.jsonl"): cls += [("inj", json.loads(l)) for l in open("/inj/data/" + f)]
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
    os.makedirs("/vol/out", exist_ok=True); json.dump(R, open(f"/vol/out/{tag}.json", "w")); vol.commit(); print("DONE", tag, R["secs"])
@app.local_entrypoint()
def main():
    fs = [run.spawn(m, t) for m, t in [("Qwen/Qwen3.6-27B", "q36_27b"), ("prism-ml/Bonsai-27B-unpacked", "bonsai27_1bit"), ("prism-ml/Ternary-Bonsai-27B-unpacked", "tbonsai27")]]
    for f in fs: f.get()
