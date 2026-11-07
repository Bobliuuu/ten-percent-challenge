"""General-capability / anti-overfit check, base vs compressed artifact, on Modal (works for any TPQ1 artifact whose base is a HF causal LM).
 - held-out general 4-way/N-way MC benchmarks the artifact was never built or calibrated on: MMLU, ARC-Challenge, HellaSwag, TruthfulQA-mc1 (zero-shot, next-token letter logits, same prompt both models)
 - neutral-text perplexity (WikiText-2 test) and mean token KL(base||cand) on the same text
 - output = per-item correctness arrays in the scorecard's R["general"] format  (scorecard.score() gates them)
Launch: modal run --detach general_remote.py --tag q3_8b --model Qwen/Qwen3-8B     (artifact on Volume tenpct-artifact:/<tag>/art)"""
import modal, os, json
V2 = os.path.expanduser("~/ten-percent/v2")
app = modal.App("tenpct-general")
img = (modal.Image.debian_slim(python_version="3.11").pip_install("torch==2.5.1", "transformers==4.56.2", "accelerate", "numpy", "safetensors", "constriction", "huggingface_hub", "datasets<4")
       .add_local_dir(V2 + "/artifact", "/app/v2/artifact", ignore=["__pycache__", "*.pyc", "results", "tests"]))
hfv = modal.Volume.from_name("tenpct-v2-hf"); outv = modal.Volume.from_name("tenpct-artifact")
INST = "The following is a multiple choice question. Answer with the letter of the correct option."

@app.function(image=img, gpu="A100-80GB", cpu=16, memory=98304, timeout=14400, volumes={"/vol": hfv, "/out": outv})
def run(tag="q3_8b", model="Qwen/Qwen3-8B", n_per=1000, ppl_tokens=40000, seed=0):
    import glob, sys, time, random, numpy as np
    os.environ.update(HF_HOME="/vol/hf"); t0 = time.time()
    import torch; from datasets import load_dataset; from transformers import AutoTokenizer, AutoModelForCausalLM
    sys.path[:0] = ["/app/v2/artifact"]; import loader as LD
    rng = random.Random(seed); LET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    def mk(q, ch, a): return dict(q=q, ch=list(ch), a=int(a))
    def sub(rows): rows = list(rows); rng.shuffle(rows); return rows[:n_per]
    sets = {}; err = {}
    def tryload(name, fn):
        try: sets[name] = sub(fn())
        except Exception as e: err[name] = repr(e)[:300]
    tryload("mmlu", lambda: [mk(r["question"], r["choices"], r["answer"]) for r in load_dataset("cais/mmlu", "all", split="test")])
    tryload("arc_challenge", lambda: [mk(r["question"], r["choices"]["text"], r["choices"]["label"].index(r["answerKey"])) for r in load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test") if r["answerKey"] in r["choices"]["label"]])
    tryload("hellaswag", lambda: [mk(r["ctx"], r["endings"], r["label"]) for r in load_dataset("Rowan/hellaswag", split="validation")])
    tryload("truthfulqa_mc1", lambda: [mk(r["question"], r["mc1_targets"]["choices"], r["mc1_targets"]["labels"].index(1)) for r in load_dataset("truthful_qa", "multiple_choice", split="validation") if len(r["mc1_targets"]["choices"]) <= 20])
    try: wt = "\n\n".join(load_dataset("wikitext", "wikitext-2-raw-v1", split="test")["text"])
    except Exception as e: err["wikitext"] = repr(e)[:300]; wt = None
    snap = [d for d in glob.glob(f"/vol/hf/hub/models--{model.replace('/', '--')}/snapshots/*") + glob.glob(f"/vol/models--{model.replace('/', '--')}/snapshots/*") if glob.glob(d + "/*.safetensors")][0]
    tok = AutoTokenizer.from_pretrained(snap); ids = [tok.encode(" " + l, add_special_tokens=False)[-1] for l in LET]
    def fmt(r): return INST + "\n\n" + r["q"].strip() + "\n" + "\n".join(f"{LET[i]}. {c.strip()}" for i, c in enumerate(r["ch"])) + "\nAnswer:"
    wids = tok(wt, return_tensors="pt").input_ids[0][:ppl_tokens] if wt else None
    @torch.no_grad()
    def mc_preds(m, rows, bs=16):
        enc = [tok.encode(fmt(r), add_special_tokens=False)[-1536:] for r in rows]; order = np.argsort([-len(e) for e in enc]); out = np.zeros(len(rows), int)
        for b in range(0, len(rows), bs):
            ix = order[b:b + bs]; mx = max(len(enc[i]) for i in ix); x = torch.full((len(ix), mx), tok.pad_token_id or 0); a = torch.zeros(len(ix), mx, dtype=torch.long)
            for j, i in enumerate(ix): x[j, mx - len(enc[i]):] = torch.tensor(enc[i]); a[j, mx - len(enc[i]):] = 1          # left pad
            pos = (a.cumsum(1) - 1).clamp(min=0)
            lg = m(input_ids=x.cuda(), attention_mask=a.cuda(), position_ids=pos.cuda()).logits[:, -1, :].float()
            for j, i in enumerate(ix): out[i] = int(lg[j, ids[:len(rows[i]["ch"])]].argmax())
        return out
    def ppl_and_dist(m, ref=None, W=1024):
        nll = 0.0; n = 0; kl = 0.0; refl = []
        with torch.no_grad():
            for k, s in enumerate(range(0, len(wids) - 1, W)):
                x = wids[s:s + W + 1][None].cuda()
                if x.shape[1] < 16: break
                l = torch.log_softmax(m(input_ids=x[:, :-1]).logits[0].float(), -1); y = x[0, 1:]
                nll += -l.gather(1, y[:, None]).sum().item(); n += len(y)
                if ref is not None: kl += (ref[k].cuda().float().exp() * (ref[k].cuda().float() - l)).sum().item()
                else: refl.append(l.half().cpu())
        return nll, n, kl, refl
    R = {"model": model, "tag": tag, "errors": err, "n_per": n_per, "benchmarks": {}}; P = {}
    base = AutoModelForCausalLM.from_pretrained(snap, torch_dtype=torch.bfloat16).cuda().eval()
    for k, rows in sets.items(): P[("base", k)] = mc_preds(base, rows); print("base", k, flush=True)
    bn, N, _, ref = ppl_and_dist(base); print("base ppl", float(np.exp(bn / N)), flush=True); del base; torch.cuda.empty_cache()
    cand, _ = LD.load_model(f"/out/{tag}/art"); 
    for k, rows in sets.items(): P[("cand", k)] = mc_preds(cand, rows); print("cand", k, flush=True)
    cn, N2, kl, _ = ppl_and_dist(cand, ref); assert N2 == N
    for k, rows in sets.items():
        y = np.array([r["a"] for r in rows]); b = P[("base", k)] == y; c = P[("cand", k)] == y
        R["benchmarks"][k] = dict(chance=float(np.mean([1 / len(r["ch"]) for r in rows])), base_correct=b.tolist(), cand_correct=c.tolist(), agree=float((P[("base", k)] == P[("cand", k)]).mean()))
    R["ppl"] = dict(base=float(np.exp(bn / N)), cand=float(np.exp(cn / N)), tokens=N, text="wikitext-2-raw test"); R["kl"] = kl / N; R["secs"] = time.time() - t0
    os.makedirs(f"/out/{tag}/general", exist_ok=True); json.dump(R, open(f"/out/{tag}/general/general_result.json", "w")); outv.commit()
    summ = {k: dict(n=len(v["base_correct"]), base=round(float(np.mean(v["base_correct"])), 4), cand=round(float(np.mean(v["cand_correct"])), 4), chance=round(v["chance"], 3), agree=round(v["agree"], 4)) for k, v in R["benchmarks"].items()}
    print("RESULT_JSON " + json.dumps(dict(summary=summ, ppl=R["ppl"], kl=R["kl"], errors=err, secs=R["secs"]))); return R["ppl"]

@app.local_entrypoint()
def main(tag: str = "q3_8b", model: str = "Qwen/Qwen3-8B"):
    print(run.spawn(tag=tag, model=model).object_id)
