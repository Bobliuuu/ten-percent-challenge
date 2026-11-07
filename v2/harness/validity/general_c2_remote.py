"""General-capability check for the FILED C2 submission artifact (Qwen2.5-7B-Instruct base, TPQ-pruned classifier artifact, volume tenpct-c2fb:/art_c2fb).
The artifact has no LM head (vocab pruned, letter head Wl), so only MC benchmarks are possible (no perplexity/KL).
Uses the artifact's own predict.py (stage-1 plain letter argmax, no memory, no bias) vs bf16 base with the SAME prompt (predict.fmt) and letter tokens " A".." N".
Benchmarks: MMLU, ARC-C, HellaSwag, TruthfulQA-mc1 (options shuffled), 1000 items each, + C2 dev (in-domain reference). Controls: RTN4 / RTN3 g128 fake-quant of the base, same prompts.
Launch: modal run --detach general_c2_remote.py"""
import modal, os, json
HERE = os.path.dirname(os.path.abspath(__file__)); R_ = HERE + "/../../"
app = modal.App("tenpct-general-c2")
img = (modal.Image.debian_slim().pip_install("torch", "transformers==4.51.3", "numpy", "safetensors", "constriction", "accelerate", "datasets<4", "huggingface_hub")
       .add_local_file(R_ + "commercial/C2/dev.jsonl", "/root/c_dev.jsonl").add_local_dir(R_ + "c2_fb/artifact_code", "/root/code"))
vol = modal.Volume.from_name("tenpct-c2fb"); hf = modal.Volume.from_name("tenpct-v2-hf")
INST = "Answer the multiple-choice question by picking the single best option."
BASE = "Qwen/Qwen2.5-7B-Instruct"

@app.function(image=img, gpu="A100-80GB", cpu=8, memory=98304, timeout=14400, volumes={"/d": vol, "/hf": hf}, env={"HF_HOME": "/hf", "HF_DATASETS_CACHE": "/tmp/dc"})
def run(n_per=1000, seed=0):
    import numpy as np, shutil, time, importlib.util, tempfile, random, torch, gc
    from datasets import load_dataset
    t0 = time.time(); rng = random.Random(seed); LET = "ABCDEFGHIJKLMN"; err = {}
    def mk(q, ch, a): return dict(task="G", text=q, names=list(ch), inst=INST, answer=int(a))
    sets = {}
    def tryload(name, fn):
        try: rows = list(fn()); rng.shuffle(rows); sets[name] = rows[:n_per]
        except Exception as e: err[name] = repr(e)[:300]
    tryload("mmlu", lambda: [mk(r["question"], r["choices"], r["answer"]) for r in load_dataset("cais/mmlu", "all", split="test")])
    tryload("arc_challenge", lambda: [mk(r["question"], r["choices"]["text"], r["choices"]["label"].index(r["answerKey"])) for r in load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test") if r["answerKey"] in r["choices"]["label"]])
    tryload("hellaswag", lambda: [mk(r["ctx"], r["endings"], r["label"]) for r in load_dataset("Rowan/hellaswag", split="validation")])
    def tqa():
        out = []
        for r in load_dataset("truthful_qa", "multiple_choice", split="validation"):
            ch = r["mc1_targets"]["choices"]
            if len(ch) <= 5: p = list(range(len(ch))); rng.shuffle(p); out.append(mk(r["question"], [ch[i] for i in p], p.index(r["mc1_targets"]["labels"].index(1))))
        return out
    tryload("truthfulqa_mc1", tqa)
    dev = [json.loads(l) for l in open("/root/c_dev.jsonl")]
    for r in dev: r.pop("tools", None); r.pop("gold_calls", None)
    sets["c2_dev_indomain"] = dev
    def imp(n, p): s = importlib.util.spec_from_file_location(n, p); m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
    def pred(LG, rows): return np.array([int(np.argmax(LG[i, :len(r["names"])])) for i, r in enumerate(rows)])
    P = {}; art = tempfile.mkdtemp() + "/art"; shutil.copytree("/d/art_c2fb", art); shutil.rmtree(art + "/__pycache__", ignore_errors=True)
    PR = imp("newp", "/root/code/predict.py"); os.environ["TENPCT_NOHEAD"] = "1"; f = PR.load(art, None)
    for k, rows in sets.items(): P[("cand", k)] = pred(f.logits(rows), rows); print("cand", k, flush=True)
    del f; gc.collect(); torch.cuda.empty_cache()
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(BASE); ids = [tok.encode(" " + l)[0] for l in LET]
    @torch.no_grad()
    def base_pred(m, rows, bs=16):
        enc = [tok.encode(PR.fmt(r))[-1024:] for r in rows]; order = np.argsort([-len(e) for e in enc]); out = np.zeros(len(rows), int)
        for b in range(0, len(rows), bs):
            ix = order[b:b + bs]; mx = max(len(enc[i]) for i in ix)
            x = torch.zeros(len(ix), mx, dtype=torch.long); a = torch.zeros(len(ix), mx, dtype=torch.long)
            for j, i in enumerate(ix): x[j, :len(enc[i])] = torch.tensor(enc[i]); a[j, :len(enc[i])] = 1
            lg = m(input_ids=x.cuda(), attention_mask=a.cuda()).logits; last = a.sum(1) - 1
            lg = lg[torch.arange(len(ix)), last.cuda()][:, ids].float().cpu().numpy()
            for j, i in enumerate(ix): out[i] = int(np.argmax(lg[j, :len(rows[i]["names"])]))
        return out
    m = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16).cuda().eval()
    for k, rows in sets.items(): P[("base", k)] = base_pred(m, rows); print("base", k, flush=True)
    ctrl = {}
    for bits in (4, 3):
        qmax = 2 ** (bits - 1) - 1
        if bits == 4: orig = {n: p.data.clone() for n, p in m.model.layers.named_parameters() if p.dim() == 2}
        else:
            for n, p in m.model.layers.named_parameters():
                if p.dim() == 2: p.data.copy_(orig[n])
        with torch.no_grad():
            for n, mod in m.model.layers.named_modules():
                if isinstance(mod, torch.nn.Linear):
                    w = mod.weight.data.float().view(mod.weight.shape[0], -1, 128); sc = (w.abs().amax(-1, keepdim=True) / qmax).clamp(min=1e-8)
                    mod.weight.data = (torch.round(w / sc).clamp(-qmax - 1, qmax) * sc).view_as(mod.weight).to(torch.bfloat16)
        ctrl[f"rtn{bits}_g128"] = {k: round(float((base_pred(m, rows) == np.array([r["answer"] for r in rows])).mean()), 4) for k, rows in sets.items()}; print("control", bits, ctrl, flush=True)
    R = {"model": BASE, "artifact": "tenpct-c2fb:/art_c2fb", "errors": err, "benchmarks": {}, "control": ctrl, "note": "stage-1 letter argmax, head off, no memory; no ppl/KL (no LM head)"}
    for k, rows in sets.items():
        y = np.array([r["answer"] for r in rows]); b = P[("base", k)] == y; c = P[("cand", k)] == y
        R["benchmarks"][k] = dict(chance=float(np.mean([1 / len(r["names"]) for r in rows])), base_correct=b.tolist(), cand_correct=c.tolist(), agree=float((P[("base", k)] == P[("cand", k)]).mean()))
    R["secs"] = time.time() - t0; os.makedirs("/d/general", exist_ok=True); json.dump(R, open("/d/general/general_c2_result.json", "w")); vol.commit()
    summ = {k: dict(n=len(v["base_correct"]), base=round(float(np.mean(v["base_correct"])), 4), cand=round(float(np.mean(v["cand_correct"])), 4), chance=round(v["chance"], 3), agree=round(v["agree"], 4)) for k, v in R["benchmarks"].items()}
    print("RESULT_JSON " + json.dumps(dict(summary=summ, control=ctrl, errors=err, secs=R["secs"]))); return summ

@app.local_entrypoint()
def main(): print(run.spawn().object_id)
