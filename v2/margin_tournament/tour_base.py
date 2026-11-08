"""Base-only margin pass on Modal (1 x A100-80GB, sequential over models). Teacher-forced label tokens, same definition as task_search/dense_core.run_ct.
python3 tour_base.py <items_tour.jsonl> <models comma list>   -> volume tenpct-margintour:/base_<model>.json  (detached)"""
import modal, json, sys, os
HERE = os.path.dirname(os.path.abspath(__file__))
app = modal.App("tenpct-margintour"); vol = modal.Volume.from_name("tenpct-margintour", create_if_missing=True); hfvol = modal.Volume.from_name("tenpct-v2-hf", create_if_missing=True)
ITEMS = sys.argv[1] if len(sys.argv) > 1 else "/nonexistent"
img = (modal.Image.debian_slim(python_version="3.11").pip_install("torch==2.4.1", "transformers==4.55.0", "numpy", "accelerate", "huggingface_hub")
       .add_local_file(ITEMS, "/code/items.jsonl") if os.path.exists(ITEMS) else modal.Image.debian_slim())
@app.function(image=img, gpu="A100-80GB", memory=64000, timeout=6 * 3600, volumes={"/vol": vol, "/hf": hfvol})
def run(models):
    import torch, numpy as np, time
    os.environ["HF_HOME"] = "/hf"
    from transformers import AutoTokenizer, AutoModelForCausalLM
    items = [json.loads(l) for l in open("/code/items.jsonl")]
    for M in models:
        t0 = time.time(); tag = M.split("/")[-1]
        tok = AutoTokenizer.from_pretrained(M); tok.padding_side = "left"
        if tok.pad_token is None: tok.pad_token = tok.eos_token
        def chat(p):
            try: return tok.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True, tokenize=False, enable_thinking=False)
            except TypeError: return tok.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True, tokenize=False)
        model = AutoModelForCausalLM.from_pretrained(M, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, device_map="cuda").eval()
        eosid = tok.convert_tokens_to_ids("<|im_end|>"); enc = []
        for it in items:
            p = tok.encode(chat(it["user"]), add_special_tokens=False)[-1500:]; a = tok.encode(it["answer"], add_special_tokens=False) + [eosid]; enc.append((p, a))
        order = np.argsort([-(len(p) + len(a)) for p, a in enc]); rows = [None] * len(items); bs = 8
        with torch.no_grad():
            for b in range(0, len(order), bs):
                idx = order[b:b + bs]; mx = max(len(enc[i][0]) + len(enc[i][1]) for i in idx)
                ids = torch.full((len(idx), mx), tok.pad_token_id); att = torch.zeros(len(idx), mx, dtype=torch.long)
                for j, i in enumerate(idx):
                    t = enc[i][0] + enc[i][1]; ids[j, :len(t)] = torch.tensor(t); att[j, :len(t)] = 1
                h = model.model(input_ids=ids.cuda(), attention_mask=att.cuda()).last_hidden_state
                for j, i in enumerate(idx):
                    p, a = enc[i]; P = len(p); lg = model.lm_head(h[j, P - 1:P - 1 + len(a)]).float(); at = torch.tensor(a).cuda()
                    gl = lg.gather(1, at[:, None])[:, 0]; lg2 = lg.clone(); lg2.scatter_(1, at[:, None], -1e9); oth, _ = lg2.max(1); mg = gl - oth
                    it = items[i]; rows[i] = dict(id=it["id"], task=it["task"], role=it["role"], src=it["src"], kind=it["kind"], correct=bool((mg > 0).all().item()), m_first=float(mg[0]), m_min=float(mg.min()))
        json.dump(dict(model=M, rows=rows, secs=time.time() - t0), open(f"/vol/base_{tag}.json", "w")); vol.commit()
        print("BASE_DONE", tag, np.mean([r["correct"] for r in rows]), round(time.time() - t0), flush=True)
        del model; torch.cuda.empty_cache()
if __name__ == "__main__":
    with modal.enable_output(), app.run(detach=True):
        c = run.spawn(sys.argv[2].split(",")); print("CALL", c.object_id, flush=True)
