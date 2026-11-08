"""INJ base probe (Modal, vLLM, greedy, chat template, one fixed prompt). Reads ONLY dev_S1 / dev_S3 (and optionally audit_S2 with --audit; QA proxy, never locked_*) from volume tenpct-inj-exam.
Saves per-item outputs + first-token margin (logsumexp logprob of tokens starting 'inj' minus tokens starting 'ben') to the volume /d/results/<model>.json -> modal volume get tenpct-inj-exam results .
Run: modal run --detach probe_base.py --models Qwen/Qwen3-8B[,..]"""
import modal, json, os
H = os.path.dirname(os.path.abspath(__file__))
app = modal.App("tenpct-inj-probe")
img = (modal.Image.debian_slim(python_version="3.12").pip_install("vllm==0.10.2", "transformers==4.56.2", "huggingface_hub<1.0", "hf_transfer")
       .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"}).add_local_file(H + "/scorer.py", "/root/scorer.py").add_local_file(H + "/prompt.py", "/root/prompt.py"))
vol = modal.Volume.from_name("tenpct-inj-exam"); hfv = modal.Volume.from_name("tenpct-v2-hf", create_if_missing=True)
MODELS = {  # name -> (gpu spec, tensor_parallel, chat kwargs, fp16 params)
    "Qwen/Qwen3-8B": ("A100-80GB", 1, {"enable_thinking": False}, 8.19e9),
    "Qwen/Qwen3-14B": ("A100-80GB", 1, {"enable_thinking": False}, 14.77e9),
    "Qwen/Qwen3-32B": ("A100-80GB", 1, {"enable_thinking": False}, 32.76e9),
    "Qwen/Qwen2.5-72B-Instruct": ("H200:2", 2, {}, 72.7e9),
    "unsloth/Llama-3.3-70B-Instruct": ("H200:2", 2, {}, 70.6e9),
    "prism-ml/Bonsai-8B-unpacked": ("A100-80GB", 1, {"enable_thinking": False}, 8.19e9),
    "Qwen/Qwen3-1.7B": ("A100-40GB", 1, {"enable_thinking": False}, 1.72e9),
    "Qwen/Qwen3-4B": ("A100-40GB", 1, {"enable_thinking": False}, 4.02e9),
}
def _run(model, files):
    suffix = "__audit" if files == ["audit_S2"] else ""
    import sys; sys.path.insert(0, "/root"); os.environ["HF_HOME"] = "/hf"
    import scorer, prompt, numpy as np
    from vllm import LLM, SamplingParams
    _, tp, kw, npar = MODELS[model]
    items = []
    for f in files: items += [json.loads(l) for l in open(f"/d/data/{f}.jsonl")]
    llm = LLM(model, dtype="bfloat16", max_model_len=4096, gpu_memory_utilization=0.92, enable_prefix_caching=False, seed=0, tensor_parallel_size=tp)
    sp = SamplingParams(temperature=0, max_tokens=16, logprobs=20)
    outs = llm.chat([[{"role": "user", "content": prompt.user_message(r["text"])}] for r in items], sp, chat_template_kwargs=kw or None, use_tqdm=False)
    rows = []
    for r, o in zip(items, outs):
        c = o.outputs[0]; m = None
        if c.logprobs:
            lp = c.logprobs[0]; a = [v.logprob for v in lp.values() if (v.decoded_token or "").strip().lower().startswith("inj")]
            b = [v.logprob for v in lp.values() if (v.decoded_token or "").strip().lower().startswith("ben")]
            if a or b: m = float(np.logaddexp.reduce(a) if a else -30) - float(np.logaddexp.reduce(b) if b else -30)
        v = scorer.parse_label(c.text)
        rows.append(dict(id=r["id"], source=r["source"], subset=r["subset"], kind=r["kind"], gold=r["gold"], out=c.text, pred=v, correct=v == r["gold"], inj_minus_ben_logit=m, out_tokens=len(c.token_ids)))
    res = dict(model=model, nparams=npar, prompt=prompt.PROMPT, summary=scorer.summarize(rows), items=rows)
    os.makedirs("/d/results_v2", exist_ok=True)
    json.dump(res, open("/d/results_v2/" + model.replace("/", "__") + suffix + ".json", "w")); vol.commit()
    return res["summary"]
_kw = dict(image=img, volumes={"/d": vol, "/hf": hfv}, timeout=3 * 3600)
@app.function(gpu="A100-80GB", **_kw)
def run_a100(model, files): return _run(model, files)
@app.function(gpu="A100-40GB", **_kw)
def run_a40(model, files): return _run(model, files)
@app.function(gpu="H200:2", **_kw)
def run_h2(model, files): return _run(model, files)
@app.local_entrypoint()
def main(models: str, audit: bool = False):
    files = ["audit_S2"] if audit else ["dev_S1", "dev_S3"]
    cs = []
    for m in models.split(","):
        g = MODELS[m][0]; fn = run_h2 if g.startswith("H200") else (run_a40 if g == "A100-40GB" else run_a100)
        cs.append((m, fn.spawn(m, files))); print("SPAWNED", m, flush=True)
    for m, c in cs: print("DONE", m, json.dumps(c.get()), flush=True)
