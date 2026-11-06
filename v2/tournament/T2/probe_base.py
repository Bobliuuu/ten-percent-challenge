"""Probe bases on T2 (PII span JSON generation). Same prompt (scorer.PROMPT_SYSTEM), same chat template, greedy, for every model.
Items: first 75 per kind (by hash order) of S1-dev and S3-dev = 300 + 300. NEVER touches *_locked files.
Run:  modal run --detach probe_base.py            (all models)   |  modal run --detach probe_base.py --only Qwen3-8B
Results: Modal volume tenpct-t2-res:/<model>.json (durable) and, if the client is still attached, results/<model>.json locally.
vLLM offline; HF cache on volume tenpct-v2-hf. Non-thinking for Qwen3 (enable_thinking=False); gpt-oss reasoning_effort=low."""
import json, os, sys, modal
HERE = os.path.dirname(os.path.abspath(__file__))
app = modal.App("tenpct-t2-probe")
hf_vol = modal.Volume.from_name("tenpct-v2-hf", create_if_missing=True)
res_vol = modal.Volume.from_name("tenpct-t2-res", create_if_missing=True)
image = (modal.Image.debian_slim(python_version="3.12").pip_install("vllm==0.11.0", "transformers==4.57.1", "huggingface_hub[hf_transfer]")
         .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HOME": "/hf", "VLLM_USE_V1": "1"})
         .add_local_file(HERE + "/scorer.py", "/root/scorer.py"))
MODELS = {  # name: (hf id, gpu, extra chat kwargs, max_new_tokens)
    "Qwen3-30B-A3B-Instruct-2507": ("Qwen/Qwen3-30B-A3B-Instruct-2507", "H100", {}, 1024),
    "Qwen2.5-32B-Instruct": ("Qwen/Qwen2.5-32B-Instruct", "H200", {}, 1024),
    "Qwen3-14B": ("Qwen/Qwen3-14B", "L40S", {"enable_thinking": False}, 1024),
    "Qwen3-8B": ("Qwen/Qwen3-8B", "L40S", {"enable_thinking": False}, 1024),
    # Llama-3.1-8B-Instruct SKIPPED: gated (manual approval), HF token 'Bobliu' gets 403 on config.json
    "gpt-oss-20b": ("openai/gpt-oss-20b", "H100", {"reasoning_effort": "low"}, 3072),
}
def run(name, items, hf_token=None):
    import scorer as S
    from vllm import LLM, SamplingParams
    hf, gpu, kw, mx = MODELS[name]
    if hf_token: os.environ["HF_TOKEN"] = hf_token
    llm = LLM(hf, dtype="bfloat16", max_model_len=6144, gpu_memory_utilization=0.95, download_dir="/hf", enforce_eager=False)
    sp = SamplingParams(temperature=0.0, max_tokens=mx, skip_special_tokens=("gpt-oss" not in name))
    msgs = [[{"role": "system", "content": S.PROMPT_SYSTEM}, {"role": "user", "content": S.user_msg(it["text"])}] for it in items]
    outs = llm.chat(msgs, sp, chat_template_kwargs=kw)
    rows = []
    for it, o in zip(items, outs):
        raw = o.outputs[0].text
        r = S.score_item(raw, it["text"], it["gold"]); r.update(id=it["id"], source=it["source"], kind=it["kind"], output=raw, finish=o.outputs[0].finish_reason)
        rows.append(r)
    summ = S.summarize(rows)
    res = dict(model=name, hf=hf, n=len(rows), summary=summ, items=rows)
    os.makedirs("/res", exist_ok=True); json.dump(res, open(f"/res/{name}.json", "w")); res_vol.commit()
    return res
@app.function(gpu=["H100","A100-80GB"], image=image, timeout=4 * 3600, volumes={"/hf": hf_vol, "/res": res_vol})
def run_h100(name, items, hf_token=None): return run(name, items, hf_token)
@app.function(gpu=os.environ.get("T2_BIG_GPU","H200,H100").split(","), image=image, timeout=4 * 3600, volumes={"/hf": hf_vol, "/res": res_vol})
def run_h200(name, items, hf_token=None): return run(name, items, hf_token)
@app.function(gpu=os.environ.get("T2_SMALL_GPU","L40S,A100-80GB,H100,A100-40GB").split(","), image=image, timeout=4 * 3600, volumes={"/hf": hf_vol, "/res": res_vol})
def run_l40s(name, items, hf_token=None): return run(name, items, hf_token)
def probe_items():
    out = []  # S1: 75/kind. S3 (gretel) has almost no PII-free texts (5 in dev), so: 100 one + 100 few + 95 many + 5 nopii
    q = {"s1_dev": dict(nopii=75, one=75, few=75, many=75), "s3_dev": dict(nopii=5, one=100, few=100, many=95)}
    for f in ["s1_dev", "s3_dev"]:
        rows = [json.loads(l) for l in open(f"{HERE}/data/{f}.jsonl")]
        for k, n in q[f].items(): out += [r for r in rows if r["kind"] == k][:n]
    return out
@app.local_entrypoint()
def main(only: str = ""):
    items = probe_items(); print("probe items", len(items))
    fns = {"H100": run_h100, "H200": run_h200, "L40S": run_l40s}
    calls = {}
    for name, (hf, gpu, _, _) in MODELS.items():
        if only and name != only: continue
        calls[name] = fns[gpu].spawn(name, items)
    os.makedirs(HERE + "/results", exist_ok=True)
    for name, c in calls.items():
        try:
            r = c.get(); json.dump(r, open(f"{HERE}/results/{name}.json", "w"), indent=1, ensure_ascii=False); print(name, r["summary"]["all"])
        except Exception as e: print(name, "FAILED", repr(e)[:300])
