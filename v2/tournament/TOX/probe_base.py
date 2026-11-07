"""TOX base probe (Modal, vLLM, greedy, chat template, one FIXED zero-shot instruction prompt = scorer.PROMPT). Reads ONLY s1_dev, s3_dev, s2_probe from volume
tenpct-tox-exam (explicit whitelist; locked.jsonl is never opened). Output = the model's GENERATED verdict word, first-token logprobs for the 5 label first-tokens saved.
Run: modal run --detach probe_base.py --models Qwen/Qwen3-8B[,..]     (client stays attached via spawn+get; results also persisted to volume /d/results/<model>.json)"""
import modal, json, os
app = modal.App("tenpct-tox-probe")
HERE = os.path.dirname(os.path.abspath(__file__))
img = (modal.Image.debian_slim(python_version="3.12").pip_install("vllm==0.10.2", "transformers==4.56.2", "huggingface_hub<1.0", "hf_transfer")
       .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"}).add_local_file(f"{HERE}/scorer.py", "/root/scorer.py"))
vol = modal.Volume.from_name("tenpct-tox-exam"); hfv = modal.Volume.from_name("tenpct-v2-hf")
KW = {"Qwen/Qwen3-8B": {"enable_thinking": False}, "Qwen/Qwen3-14B": {"enable_thinking": False}, "Qwen/Qwen3-32B": {"enable_thinking": False}}
TP = {"Qwen/Qwen2.5-72B-Instruct": 4, "unsloth/Llama-3.3-70B-Instruct": 4}
DEV_FILES = ("s1_dev.jsonl", "s3_dev.jsonl", "s2_probe.jsonl")     # whitelist: NO locked file
def _run(model):
    import sys; sys.path.insert(0, "/root"); os.environ["HF_HOME"] = "/hf"
    import scorer
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model)
    ft = {l: tok.encode(l, add_special_tokens=False)[0] for l in scorer.LABELS}
    assert len(set(ft.values())) == 5, ft
    items = []
    for f in DEV_FILES: items += [json.loads(l) for l in open(f"/d/data/{f}")]
    llm = LLM(model, dtype="bfloat16", max_model_len=3072, gpu_memory_utilization=0.92, enable_prefix_caching=False, seed=0, tensor_parallel_size=TP.get(model, 1))
    sp = SamplingParams(temperature=0, max_tokens=16, logprobs=20)
    msgs = [[{"role": "user", "content": scorer.PROMPT.format(t=r["text"])}] for r in items]
    outs = llm.chat(msgs, sp, chat_template_kwargs=KW.get(model) or None, use_tqdm=False)
    rows = []
    for r, o in zip(items, outs):
        c = o.outputs[0]; lp = {l: -30.0 for l in scorer.LABELS}
        if c.logprobs:
            for l, tid in ft.items():
                if tid in c.logprobs[0]: lp[l] = float(c.logprobs[0][tid].logprob)
        rows.append(dict(id=r["id"], split=("s1_dev" if r["source"] == "S1" else "s3_dev" if r["source"] == "S3" else "s2_probe"), source=r["source"], kind=r["kind"], gold=r["gold"], hard=r["hard"],
                         group=r["group"], out=c.text, pred=scorer.parse_verdict(c.text), lp=lp, out_tokens=len(c.token_ids)))
    res = dict(model=model, n=len(rows), first_tokens=ft, items=rows, summary=scorer.summarize(rows))
    os.makedirs("/d/results", exist_ok=True); json.dump(res, open("/d/results/" + model.replace("/", "__") + ".json", "w")); vol.commit()
    return res["summary"]
GPUS = {1: ["A100-80GB", "H100"], 4: "A100-80GB:4"}
_kw = dict(image=img, volumes={"/d": vol, "/hf": hfv}, timeout=3 * 3600)
@app.function(gpu=GPUS[1], **_kw)
def run_1(model): return _run(model)
@app.function(gpu=GPUS[4], **_kw)
def run_4(model): return _run(model)
@app.local_entrypoint()
def main(models: str):
    for m in models.split(","):
        c = (run_4 if TP.get(m, 1) == 4 else run_1).spawn(m); print("SPAWNED", m, c.object_id, flush=True)
        print("DONE", m, json.dumps({k: round(v["acc"], 3) for k, v in c.get().items()}), flush=True)
