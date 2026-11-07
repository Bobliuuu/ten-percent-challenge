"""Big-MoE base probe: adapted from tournament/T1/probe_base.py. multi-GPU vllm TP. One model per run:
   modal run --detach probe_big.py --only Qwen3-235B-FP8
Results -> Modal volume tenpct-l2-res:/<name>.json ; fetch with `modal volume get tenpct-l2-res / results/`."""
import json, os, modal
HERE = os.path.dirname(os.path.abspath(__file__))
app = modal.App("tenpct-l2-base-probe")
hf = modal.Volume.from_name("tenpct-v2-hf", create_if_missing=True)
res = modal.Volume.from_name("tenpct-l2-res", create_if_missing=True)
img = (modal.Image.debian_slim(python_version="3.12").pip_install("vllm==0.11.0", "transformers==4.57.1", "huggingface_hub", "hf_transfer")
       .env({"HF_HOME": "/hf", "HF_HUB_ENABLE_HF_TRANSFER": "1", "VLLM_USE_V1": "1"})
       .add_local_file(f"{HERE}/scorer.py", "/app/scorer.py").add_local_file(f"{HERE}/probe_items.jsonl", "/app/probe_items.jsonl"))
# name -> (repo, gpu, tp, chat kwargs, max_tokens, precision, extra LLM kwargs)
M = {
 "Qwen3-235B-FP8": ("Qwen/Qwen3-235B-A22B-Instruct-2507-FP8", "H100:4", 4, {}, 1024, "fp8", {}),
 "gpt-oss-120b": ("openai/gpt-oss-120b", "H100:2", 2, {"reasoning_effort": "low"}, 3000, "mxfp4", {}),
 "GLM-4.5-Air-FP8": ("zai-org/GLM-4.5-Air-FP8", "H100:2", 2, {"enable_thinking": False}, 1024, "fp8", {}),
 "GLM-4.5-Air": ("zai-org/GLM-4.5-Air", "H100:4", 4, {"enable_thinking": False}, 1024, "bf16", {}),
 "Qwen3-Coder-480B-FP8": ("Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8", "H100:8", 8, {}, 1024, "fp8", {"enable_expert_parallel": True}),
 "GLM-4.5-FP8": ("zai-org/GLM-4.5-FP8", "H100:8", 8, {"enable_thinking": False}, 1024, "fp8", {}),
 "Qwen3-Next-80B": ("Qwen/Qwen3-Next-80B-A3B-Instruct", "H100:2", 2, {}, 1024, "bf16", {}),
 "Mixtral-8x22B": ("mistralai/Mixtral-8x22B-Instruct-v0.1", "H100:4", 4, {}, 1024, "bf16", {}),
 "Llama-4-Scout": ("meta-llama/Llama-4-Scout-17B-16E-Instruct", "H100:2", 2, {}, 1024, "bf16", {}),
 "Llama-4-Maverick-FP8": ("meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8", "H100:8", 8, {}, 1024, "fp8", {}),
}

def _run(name):
    import sys, time
    sys.path.insert(0, "/app")
    from scorer import score_item
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    repo, _, tp, kw, mt, prec, extra = M[name]
    items = [json.loads(l) for l in open("/app/probe_items.jsonl")]
    tok = AutoTokenizer.from_pretrained(repo)
    prompts = []
    for it in items:
        tools = [{"type": "function", "function": t} for t in it["tools"]]
        prompts.append(tok.apply_chat_template([{"role": "user", "content": it["query"]}], tools=tools, add_generation_prompt=True, tokenize=False, **kw))
    t0 = time.time()
    llm = LLM(repo, tensor_parallel_size=tp, max_model_len=20000, gpu_memory_utilization=0.85, max_num_seqs=128, enable_prefix_caching=True, seed=0, **extra)
    tload = time.time() - t0
    outs = llm.generate(prompts, SamplingParams(temperature=0, max_tokens=mt, skip_special_tokens=False), use_tqdm=False)
    rows = []
    for it, p, o in zip(items, prompts, outs):
        txt = o.outputs[0].text
        s = score_item(txt, it)
        rows.append(dict(id=it["id"], source=it["source"], kind=it["kind"], correct=bool(s["correct"]), parse_fail=bool(s["parse_fail"]), n_calls=s["n_calls"],
                         finish=o.outputs[0].finish_reason, n_prompt_tokens=len(o.prompt_token_ids), output=txt))
    out = dict(model=name, repo=repo, precision=prec, gpu=M[name][1], template_kwargs=kw, max_tokens=mt, secs=time.time() - t0, load_secs=tload, prompt_example=prompts[0], rows=rows)
    json.dump(out, open(f"/res/{name}.json", "w")); res.commit()
    return out

FN = {}
def _mk(name):
    def f(): return _run(name)
    f.__name__ = f.__qualname__ = "probe_" + name.replace("-", "_").replace(".", "_")
    globals()[f.__name__] = f
    return app.function(image=img, gpu=M[name][1], volumes={"/hf": hf, "/res": res}, timeout=4 * 3600,
                        secrets=[modal.Secret.from_name("huggingface-secret")] if os.environ.get("HFSEC") else [])(f)
for _n in M: FN[_n] = _mk(_n)

@app.local_entrypoint()
def main(only: str):
    calls = {n: FN[n].spawn() for n in only.split(",")}
    os.makedirs(f"{HERE}/results", exist_ok=True)
    for n, c in calls.items():
        try:
            r = c.get(); json.dump(r, open(f"{HERE}/results/{n}.json", "w")); print("done", n, sum(x["correct"] for x in r["rows"]) / len(r["rows"]))
        except Exception as e: print("FAILED", n, repr(e)[:800])
