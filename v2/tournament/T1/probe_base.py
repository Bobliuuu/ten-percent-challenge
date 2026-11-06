"""Probe bases on 300 S1-dev + 300 S3-dev items (100 per kind per source, first by hash order), greedy, base's own chat template
with tools=[...] (native tool format), single user message, no system prompt. Same scorer for all.
Run:  modal run --detach probe_base.py            (all models, parallel)
      modal run --detach probe_base.py --only Qwen3-8B
Results: Modal volume tenpct-t1-res:/<name>.json (+ local copy by entrypoint if client alive; otherwise `modal volume get tenpct-t1-res / results/`).
NEVER reads data/locked.jsonl."""
import json, os, modal
HERE = os.path.dirname(os.path.abspath(__file__))
app = modal.App("tenpct-t1-probe")
hf = modal.Volume.from_name("tenpct-v2-hf", create_if_missing=True)
res = modal.Volume.from_name("tenpct-t1-res", create_if_missing=True)
img = (modal.Image.debian_slim(python_version="3.12").pip_install("vllm==0.11.0", "transformers==4.57.1", "huggingface_hub", "hf_transfer")
       .env({"HF_HOME": "/hf", "HF_HUB_ENABLE_HF_TRANSFER": "1", "VLLM_USE_V1": "1"})
       .add_local_file(f"{HERE}/scorer.py", "/app/scorer.py").add_local_file(f"{HERE}/probe_items.jsonl", "/app/probe_items.jsonl"))
# name -> (hf repo, gpu, extra chat-template kwargs, max_tokens, note)
MODELS = {
    "Qwen3-30B-A3B-Instruct-2507": ("Qwen/Qwen3-30B-A3B-Instruct-2507", "A100-80GB", {}, 1024),
    "Qwen2.5-32B-Instruct": ("Qwen/Qwen2.5-32B-Instruct", "A100-80GB", {}, 1024),
    "Qwen3-14B": ("Qwen/Qwen3-14B", "A100-40GB", {"enable_thinking": False}, 1024),
    "Qwen3-8B": ("Qwen/Qwen3-8B", "A100-40GB", {"enable_thinking": False}, 1024),
    "Llama-3.1-8B-Instruct": ("unsloth/Llama-3.1-8B-Instruct", "A100-40GB", {}, 1024),  # meta-llama repo is gated (no access); unsloth = ungated byte-identical mirror
    "gpt-oss-20b": ("openai/gpt-oss-20b", "A100-40GB", {"reasoning_effort": "low"}, 3000),
}

def _run(name):
    import sys, time, collections
    sys.path.insert(0, "/app")
    from scorer import score_item
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    repo, _, kw, mt = MODELS[name]
    items = [json.loads(l) for l in open("/app/probe_items.jsonl")]
    tok = AutoTokenizer.from_pretrained(repo)
    prompts = []
    for it in items:
        tools = [{"type": "function", "function": t} for t in it["tools"]]
        prompts.append(tok.apply_chat_template([{"role": "user", "content": it["query"]}], tools=tools, add_generation_prompt=True, tokenize=False, **kw))
    t0 = time.time()
    llm = LLM(repo, max_model_len=20000, gpu_memory_utilization=0.92, enable_prefix_caching=True, seed=0)
    outs = llm.generate(prompts, SamplingParams(temperature=0, max_tokens=mt, skip_special_tokens=False), use_tqdm=False)
    rows = []
    for it, p, o in zip(items, prompts, outs):
        txt = o.outputs[0].text
        s = score_item(txt, it)
        rows.append(dict(id=it["id"], source=it["source"], kind=it["kind"], correct=bool(s["correct"]), parse_fail=bool(s["parse_fail"]), n_calls=s["n_calls"],
                         finish=o.outputs[0].finish_reason, n_prompt_tokens=len(o.prompt_token_ids), output=txt))
    out = dict(model=name, repo=repo, template_kwargs=kw, max_tokens=mt, secs=time.time() - t0, prompt_example=prompts[0], rows=rows)
    json.dump(out, open(f"/res/{name}.json", "w")); res.commit()
    return out

@app.function(image=img, gpu=["H100","A100-80GB"], volumes={"/hf": hf, "/res": res}, timeout=3 * 3600)
def probe_Qwen3_30B_A3B_Instruct_2507(): return _run("Qwen3-30B-A3B-Instruct-2507")

@app.function(image=img, gpu=["H100","A100-80GB"], volumes={"/hf": hf, "/res": res}, timeout=3 * 3600)
def probe_Qwen2_5_32B_Instruct(): return _run("Qwen2.5-32B-Instruct")

@app.function(image=img, gpu=["A100-40GB","A100-80GB","L40S","H100"], volumes={"/hf": hf, "/res": res}, timeout=3 * 3600)
def probe_Qwen3_14B(): return _run("Qwen3-14B")

@app.function(image=img, gpu=["A100-40GB","A100-80GB","L40S","H100"], volumes={"/hf": hf, "/res": res}, timeout=3 * 3600)
def probe_Qwen3_8B(): return _run("Qwen3-8B")

@app.function(image=img, gpu=["A100-40GB","A100-80GB","L40S","H100"], volumes={"/hf": hf, "/res": res}, timeout=3 * 3600)
def probe_Llama_3_1_8B_Instruct(): return _run("Llama-3.1-8B-Instruct")

@app.function(image=img, gpu=["A100-40GB","A100-80GB","L40S","H100"], volumes={"/hf": hf, "/res": res}, timeout=3 * 3600)
def probe_gpt_oss_20b(): return _run("gpt-oss-20b")

FN = {"Qwen3-30B-A3B-Instruct-2507": probe_Qwen3_30B_A3B_Instruct_2507, "Qwen2.5-32B-Instruct": probe_Qwen2_5_32B_Instruct, "Qwen3-14B": probe_Qwen3_14B, "Qwen3-8B": probe_Qwen3_8B, "Llama-3.1-8B-Instruct": probe_Llama_3_1_8B_Instruct, "gpt-oss-20b": probe_gpt_oss_20b}

@app.local_entrypoint()
def main(only: str = ""):
    names = [n for n in MODELS if not only or n in only.split(",")]
    calls = {n: FN[n].spawn() for n in names}
    os.makedirs(f"{HERE}/results", exist_ok=True)
    for n, c in calls.items():
        try:
            r = c.get(); json.dump(r, open(f"{HERE}/results/{n}.json", "w")); print("done", n, sum(x["correct"] for x in r["rows"]) / len(r["rows"]))
        except Exception as e: print("FAILED", n, repr(e)[:500])
