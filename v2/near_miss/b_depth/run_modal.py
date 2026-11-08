"""python -m modal run --detach run_modal.py --jobfile jobs_7b.json   (detached; results on Volume tenpct-nearmiss-b + RESULT_JSON in logs)"""
import modal, json, os
HERE = os.path.dirname(os.path.abspath(__file__))
app = modal.App("tenpct-nm-b-depth")
vol = modal.Volume.from_name("tenpct-nearmiss-b", create_if_missing=True)
hf = modal.Volume.from_name("tenpct-v2-hf", create_if_missing=True)
img = (modal.Image.debian_slim(python_version="3.11").pip_install("vllm==0.11.0", "transformers==4.56.2", "huggingface_hub", "hf_transfer", "safetensors", "accelerate")
       .add_local_dir(HERE + "/data", "/code/data").add_local_file(HERE + "/core.py", "/code/core.py").add_local_file(HERE + "/gen.py", "/code/gen.py"))
@app.function(image=img, gpu="A100-80GB", memory=100000, timeout=8 * 3600, volumes={"/vol": vol, "/hf": hf}, ephemeral_disk=524288)
def job(j):
    import sys, subprocess, shutil, torch, os
    os.environ["HF_HOME"] = "/hf"; sys.path.insert(0, "/code"); import core
    from transformers import AutoTokenizer
    model = j["model"]; tag = model.split("/")[-1]; path = core.snap(model); hf.commit(); tok = AutoTokenizer.from_pretrained(path)
    cal, TOXP, INJP = core.load_texts(tok); conf = json.load(open(path + "/config.json"))
    sf = f"/vol/{tag}_sens.json"
    if os.path.exists(sf): sens = json.load(open(sf))
    else:
        sens = core.sensitivity(model, path, tok, cal, TOXP, INJP); json.dump(sens, open(sf, "w")); vol.commit()
    print("SENS", json.dumps(sens), flush=True)
    keep = core.keep_vocab(tok, cal, TOXP, INJP, conf["vocab_size"]); json.dump(keep, open(f"/vol/{tag}_keep.json", "w")); print("KEEP", len(keep), flush=True)
    ws = None
    if any(c.get("ffn") or c.get("grp") for c in j["cfgs"]):
        wf = f"/vol/{tag}_width.pt"
        if os.path.exists(wf): ws = torch.load(wf)
        else: ws = core.width_stats(path, tok, cal, TOXP, INJP); torch.save(ws, wf); vol.commit()
    for cfg in j["cfgs"]:
        name = f"{tag}_{cfg['name']}"
        if os.path.exists(f"/vol/{name}.gen.json"): continue
        t0 = time.time() if False else 0
        od = "/work/m"; shutil.rmtree(od, ignore_errors=True)
        by = core.build(path, cfg, sens, keep, od, tok, ws); torch.cuda.empty_cache()
        kf = f"/vol/{tag}_keep.json" if cfg.get("vtrim") else None
        r = subprocess.run([sys.executable, "/code/gen.py", od, f"/vol/{name}.gen.json"] + ([kf] if kf else []), capture_output=True, text=True)
        print((r.stdout or "")[-600:], (r.stderr or "")[-1500:] if r.returncode else "", flush=True)
        res = dict(name=name, cfg=cfg, bytes=by, ok=r.returncode == 0); json.dump(res, open(f"/vol/{name}.bytes.json", "w")); vol.commit()
        print("RESULT_JSON", json.dumps(res), flush=True); shutil.rmtree(od, ignore_errors=True)
@app.local_entrypoint()
def main(jobfile: str):
    j = json.load(open(jobfile)); print(job.spawn(j).object_id)
