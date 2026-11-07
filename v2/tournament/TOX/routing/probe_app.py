"""Routing base probe (Modal, vLLM, greedy, chat template with thinking off). Deploy: modal deploy probe_app.py ; launch: python3 launch.py
Prompt: fixed instruction + queue menu (names + one-line descriptions) + ticket text; model must reply with exactly the queue name. Output = generated string.
Results -> Volume tenpct-tox-routing:/results/<model>.json (also RESULT_JSON in app logs)."""
import modal, json, os
H = os.path.dirname(os.path.abspath(__file__))
app = modal.App("tenpct-tox-routing-probe")
img = (modal.Image.debian_slim(python_version="3.12").pip_install("vllm==0.10.2", "transformers==4.56.2", "huggingface_hub<1.0", "hf_transfer")
       .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"}).add_local_file(H + "/data/probe_dev.jsonl", "/root/probe_dev.jsonl").add_local_file(H + "/taxonomy.json", "/root/taxonomy.json"))
vol = modal.Volume.from_name("tenpct-tox-routing", create_if_missing=True); hfv = modal.Volume.from_name("tenpct-v2-hf")
MODELS = {"Qwen/Qwen3-8B": 8.19e9, "Qwen/Qwen3-14B": 14.77e9, "Qwen/Qwen3-32B": 32.76e9}
PROMPT = ("You are a support-ticket router. Route the incoming message to exactly one queue from the list below.\n\nQueues:\n{MENU}\n\n"
          "Reply with exactly the queue name and nothing else.\n\nMessage:\n\"\"\"\n{T}\n\"\"\"")

_kw = dict(image=img, volumes={"/d": vol, "/hf": hfv}, timeout=3 * 3600)
@app.function(gpu=["L40S", "A100-40GB", "A100-80GB", "H100"], **_kw)
def run_small(model): return _run(model)
@app.function(gpu=["A100-80GB", "H100", "H200"], **_kw)
def run_big(model): return _run(model)
def _run(model):
    os.environ["HF_HOME"] = "/hf"
    import numpy as np
    from vllm import LLM, SamplingParams
    tax = json.load(open("/root/taxonomy.json")); items = [json.loads(l) for l in open("/root/probe_dev.jsonl")]
    menus = {"U": tax["QUEUES"], "IT": tax["IT_Q"]}
    mtxt = {k: "\n".join(f"- {q}: {d}" for q, d in v.items()) for k, v in menus.items()}
    llm = LLM(model, dtype="bfloat16", max_model_len=3072, gpu_memory_utilization=0.92, enable_prefix_caching=True, seed=0)
    sp = SamplingParams(temperature=0, max_tokens=16, logprobs=20)
    msgs = []
    for r in items:
        mk = "IT" if r["source"] == "S4" else "U"; r["menu"] = mk
        msgs.append([{"role": "user", "content": PROMPT.format(MENU=mtxt[mk], T=r["text"])}])
    outs = llm.chat(msgs, sp, chat_template_kwargs={"enable_thinking": False}, use_tqdm=False)
    tok = llm.get_tokenizer(); rows = []
    for r, o in zip(items, outs):
        c = o.outputs[0]; names = list(menus[r["menu"]]); t = c.text.strip().lower().strip("`'\" .*")
        pred = t if t in names else next((n for n in names if t.startswith(n)), None)
        # first-token margin: best first-token logprob among queue names' first tokens, gold vs best other
        lp = c.logprobs[0] if c.logprobs else {}
        first = {n: tok.encode(n, add_special_tokens=False)[0] for n in names}
        sc = {}
        for n, tid in first.items():
            sc[n] = lp[tid].logprob if tid in lp else -30.0
        s_gold = sc.get(r["queue"], -30.0); s_oth = max(v for n, v in sc.items() if n != r["queue"])
        rows.append(dict(id=r["id"], source=r["source"], queue=r["queue"], orig=r["orig"], out=c.text, pred=pred, correct=pred == r["queue"], margin=float(s_gold - s_oth), top1_lp=float(max(sc.values()))))
    res = dict(model=model, n=len(rows), rows=rows)
    os.makedirs("/d/results", exist_ok=True); json.dump(res, open("/d/results/" + model.replace("/", "__") + ".json", "w")); vol.commit()
    acc = {s: float(np.mean([x["correct"] for x in rows if x["source"] == s])) for s in sorted({x["source"] for x in rows})}
    print("RESULT_JSON", json.dumps(dict(model=model, acc=acc, unparsed=sum(x["pred"] is None for x in rows))), flush=True)
    return acc
