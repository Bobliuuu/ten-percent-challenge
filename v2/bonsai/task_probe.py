"""Bonsai-8B task probe (DEV data only). Real greedy generation via the SHIPPED pack+loader (tenpct-native:/bonsai8b_art). Strict parse: whole output == gold (case-insens., optional '.').
Deploy+spawn:  python3 task_probe.py   (spawns detached). Result: Modal volume tenpct-native:/bonsai_taskprobe.json (+ pulled to results/)."""
import modal, os, json
HERE = os.path.dirname(os.path.abspath(__file__))
app = modal.App("tenpct-bonsai-probe")
img = (modal.Image.debian_slim(python_version="3.11").pip_install("torch==2.5.1", "transformers==4.56.2", "accelerate", "numpy", "safetensors")
       .add_local_file(HERE + "/probe_items.jsonl", "/app/probe_items.jsonl"))
vol = modal.Volume.from_name("tenpct-native")
@app.function(image=img, gpu="A100-80GB", cpu=8, memory=65536, timeout=3 * 3600, volumes={"/vol": vol})
def run():
    import sys, re, time, collections, torch
    from transformers import AutoTokenizer
    sys.path.insert(0, "/vol/bonsai8b_art"); import bonsai_loader as BL
    d = "/vol/bonsai8b_art"; tok = AutoTokenizer.from_pretrained(d); tok.padding_side = "left"
    m, miss = BL.load(d); items = [json.loads(l) for l in open("/app/probe_items.jsonl")]
    pr = [tok.apply_chat_template([{"role": "user", "content": r["user"]}], add_generation_prompt=True, tokenize=False, enable_thinking=False) for r in items]
    order = sorted(range(len(items)), key=lambda i: len(pr[i])); out = {}; t0 = time.time()
    for b in range(0, len(order), 24):
        idx = order[b:b + 24]; enc = tok([pr[i] for i in idx], return_tensors="pt", padding=True, add_special_tokens=False, return_token_type_ids=False).to("cuda")
        with torch.no_grad(): g = m.generate(**enc, max_new_tokens=14, do_sample=False, pad_token_id=tok.pad_token_id)
        for j, i in enumerate(idx): out[i] = tok.decode(g[j, enc.input_ids.shape[1]:], skip_special_tokens=True)
    rows = []
    for i, r in enumerate(items):
        s = out[i]; s2 = re.sub(r"^\s*<think>\s*</think>\s*", "", s).strip()
        ok = re.fullmatch(re.escape(r["answer"]) + r"\.?", s2, re.I) is not None
        rows.append(dict(id=r["id"], task=r["task"], src=r["src"], kind=r["kind"], gold=r["answer"], out=s, ok=int(ok)))
    json.dump(dict(rows=rows, secs=time.time() - t0), open("/vol/bonsai_taskprobe.json", "w")); vol.commit()
    acc = collections.defaultdict(list)
    for r in rows: acc[r["task"]].append(r["ok"])
    print("RESULT", {k: round(sum(v) / len(v), 3) for k, v in acc.items()}, flush=True)
if __name__ == "__main__":
    with modal.enable_output(), app.run(detach=True):
        c = run.spawn(); print("CALL", c.object_id, flush=True)
