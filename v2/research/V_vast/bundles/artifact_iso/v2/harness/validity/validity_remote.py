"""Validity / anti-overfitting suite on Modal. Artifact (real predict.py API, stage-1 = plain letter-logit argmax) vs the real
Qwen2.5-7B-Instruct base, SAME prompt, SAME rows. No weights trained, no locked-exam use.
 (A) general MC benchmarks the artifact was never built on: MMLU, ARC-Challenge, HellaSwag, TruthfulQA-mc1
 (B) C2 shift on DEV (locked is spent): tool-option order shuffles x3 (cand-base gap must not depend on order)
Launch: modal run --detach validity_remote.py"""
import modal, json, os
HERE = os.path.dirname(os.path.abspath(__file__)); R = HERE + "/../../"
app = modal.App("tenpct-validity")
img = (modal.Image.debian_slim().pip_install("torch", "transformers==4.51.3", "numpy", "safetensors", "constriction", "accelerate", "datasets", "huggingface_hub")
       .add_local_file(R + "commercial/C2/dev.jsonl", "/root/c_dev.jsonl").add_local_dir(R + "c2_fb/artifact_code", "/root/code"))
vol = modal.Volume.from_name("tenpct-c2fb", create_if_missing=False); hf = modal.Volume.from_name("tenpct-v2-hf", create_if_missing=True)
INST = "Answer the multiple-choice question by picking the single best option."
BASE = "Qwen/Qwen2.5-7B-Instruct"

@app.function(image=img, gpu="A100-80GB", cpu=8, memory=98304, timeout=21600, volumes={"/d": vol, "/hf": hf}, env={"HF_HOME": "/hf"})
def run(n_per=1500):
    import numpy as np, shutil, time, importlib.util, tempfile, random, torch, gc
    from datasets import load_dataset
    t0 = time.time(); rng = random.Random(0); LET = "ABCDEFGHIJKLMN"
    def mk(q, ch, a, task="G"): return dict(task=task, text=q, names=list(ch), inst=INST, answer=int(a))
    sets = {}
    def sub(rows): rng.shuffle(rows); return rows[:n_per]
    try:
        d = load_dataset("cais/mmlu", "all", split="test"); sets["mmlu"] = sub([mk(r["question"], r["choices"], r["answer"]) for r in d])
    except Exception as e: sets["mmlu_ERR"] = str(e)
    try:
        d = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test"); ks = "ABCDE"
        sets["arc_c"] = sub([mk(r["question"], r["choices"]["text"], r["choices"]["label"].index(r["answerKey"])) for r in d if r["answerKey"] in r["choices"]["label"]])
    except Exception as e: sets["arc_c_ERR"] = str(e)
    try:
        d = load_dataset("Rowan/hellaswag", split="validation"); sets["hellaswag"] = sub([mk(r["ctx"], r["endings"], r["label"]) for r in d])
    except Exception as e: sets["hellaswag_ERR"] = str(e)
    try:
        d = load_dataset("truthful_qa", "multiple_choice", split="validation")
        sets["truthfulqa_mc1"] = sub([mk(r["question"], r["mc1_targets"]["choices"], r["mc1_targets"]["labels"].index(1)) for r in d if len(r["mc1_targets"]["choices"]) <= 14])
    except Exception as e: sets["truthfulqa_ERR"] = str(e)
    dev = [json.loads(l) for l in open("/root/c_dev.jsonl")]
    for r in dev: r.pop("tools", None); r.pop("gold_calls", None)
    def shuf(rows, seed):
        g = random.Random(seed); out = []
        for r in rows:
            n = len(r["names"]); none = r["names"][-1].startswith("None of these"); k = n - 1 if none else n
            p = list(range(k)); g.shuffle(p); names = [r["names"][i] for i in p] + (r["names"][-1:] if none else [])
            a = r["answer"]; na = p.index(a) if a < k else a
            out.append(dict(r, names=names, answer=na))
        return out
    sets["c2_dev_orig"] = dev
    for s in (1, 2, 3): sets[f"c2_dev_shuf{s}"] = shuf(dev, s)
    sets = {k: v for k, v in sets.items() if not k.endswith("_ERR")} | {k: v for k, v in sets.items() if k.endswith("_ERR")}
    def pred_from(LG, rows): return [int(np.argmax(LG[i, :len(r["names"])])) for i, r in enumerate(rows)]
    def imp(n, p): s = importlib.util.spec_from_file_location(n, p); m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
    res = {k: v for k, v in sets.items() if k.endswith("_ERR")}; P = {}
    # ---- artifact
    art = tempfile.mkdtemp() + "/art"; shutil.copytree("/d/art_c2fb", art); shutil.rmtree(art + "/__pycache__", ignore_errors=True)
    PR = imp("newp", "/root/code/predict.py"); f = PR.load(art, None)
    for k, rows in sets.items():
        if k.endswith("_ERR"): continue
        LG = f.logits(rows); P[("cand", k)] = pred_from(LG, rows); print("cand", k, flush=True)
    del f; gc.collect(); torch.cuda.empty_cache()
    # ---- base (same fmt, same letter tokens " A".." N", full-precision bf16 7B)
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(BASE); m = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16).cuda().eval()
    ids = [tok.encode(" " + l)[0] for l in LET]
    @torch.no_grad()
    def base_logits(rows, bs=16):
        enc = [tok.encode(PR.fmt(r))[-1024:] for r in rows]; order = np.argsort([-len(e) for e in enc]); out = np.zeros((len(rows), 14), np.float32)
        for b in range(0, len(rows), bs):
            ix = order[b:b + bs]; mx = max(len(enc[i]) for i in ix)
            x = torch.zeros(len(ix), mx, dtype=torch.long); a = torch.zeros(len(ix), mx, dtype=torch.long)
            for j, i in enumerate(ix): x[j, :len(enc[i])] = torch.tensor(enc[i]); a[j, :len(enc[i])] = 1
            lg = m(input_ids=x.cuda(), attention_mask=a.cuda()).logits; last = a.sum(1) - 1
            out[ix] = lg[torch.arange(len(ix)), last.cuda()][:, ids].float().cpu().numpy()
        return out
    for k, rows in sets.items():
        if k.endswith("_ERR"): continue
        P[("base", k)] = pred_from(base_logits(rows), rows); print("base", k, flush=True)
    # ---- stats
    rg = np.random.default_rng(0)
    for k, rows in sets.items():
        if k.endswith("_ERR"): continue
        y = np.array([r["answer"] for r in rows]); b = np.array(P[("base", k)]) == y; c = np.array(P[("cand", k)]) == y; d = c.astype(int) - b.astype(int)
        bs = np.sort([d[rg.integers(0, len(d), len(d))].mean() for _ in range(2000)])
        chance = float(np.mean([1 / len(r["names"]) for r in rows]))
        res[k] = dict(n=len(rows), chance=round(chance, 4), base=round(float(b.mean()), 4), cand=round(float(c.mean()), 4), diff=round(float(d.mean()), 4),
                      diff_ci95=[round(float(bs[50]), 4), round(float(bs[1949]), 4)], agree=round(float((np.array(P[("base", k)]) == np.array(P[("cand", k)])).mean()), 4),
                      b_only=int(((b) & (~c)).sum()), c_only=int(((~b) & (c)).sum()))
    res["secs"] = time.time() - t0
    os.makedirs("/d/validity", exist_ok=True); json.dump(res, open("/d/validity/result.json", "w"), indent=1); vol.commit(); print("RESULT_JSON " + json.dumps(res)); return res

@app.local_entrypoint()
def main():
    print(run.spawn().object_id)
