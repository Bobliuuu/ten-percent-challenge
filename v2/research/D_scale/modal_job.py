"""D_scale on Modal: one job = [quantise Qwen2.5-72B-Instruct variant] -> [export fp8-dynamic HF ckpt] -> [vLLM T1 (1300 dev items) + label tasks (5414)] on 1 H200.
launch: python modal_job.py variants.json   (spawns detached; results -> Modal volume tenpct-dscale + local ./modal_results/)"""
import modal, json, sys, os
app = modal.App("tenpct-dscale")
vol = modal.Volume.from_name("tenpct-dscale", create_if_missing=True)
HERE = os.path.dirname(os.path.abspath(__file__)) + "/modal"
img = (modal.Image.debian_slim(python_version="3.11").pip_install("vllm==0.11.0", "transformers==4.57.1", "huggingface_hub<1.0", "safetensors", "numpy<2.3")
       .add_local_dir(HERE, "/code"))

@app.function(image=img, gpu="H200", memory=220000, ephemeral_disk=900 * 1024, timeout=3 * 3600, volumes={"/vol": vol}, max_containers=6)
def run(var):
    import subprocess, shutil, glob, time
    t0 = time.time(); name = var["name"]; W = "/work"; os.makedirs(W + "/hf/Qwen2.5-72B", exist_ok=True)
    for f in ["dense_stream.py", "export_hf.py", "vllm_base.py", "vllm_ct.py", "dense_data.json", "calib_ct.jsonl", "items.jsonl", "qcfg.json"]: shutil.copy("/code/" + f, W + "/" + f)
    shutil.copytree("/code/t1v2", W + "/t1v2", dirs_exist_ok=True); shutil.copytree("/code/t1v3", W + "/t1v3", dirs_exist_ok=True); shutil.copy("/code/sens_g128.json", W + "/hf/Qwen2.5-72B/sens_g128.json")
    os.environ.update(HF_HOME="/hf", DS_WORK=W, DENSE_T1=W + "/t1v2", DENSE_DATA=W + "/dense_data.json", PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    from huggingface_hub import snapshot_download
    P = snapshot_download("Qwen/Qwen2.5-72B-Instruct", allow_patterns=["*.json", "*.safetensors", "*.txt"]); print("downloaded", time.time() - t0, flush=True)
    cfg = dict(model_dir=P, tag="Qwen2.5-72B", mode="quant", target_ratio=0.096, emb_L=4, head_L=4, g=128); cfg.update(var.get("cfg", {})); cfg["name"] = name
    if cfg.get("calib_ct"): cfg["calib_ct"] = W + "/calib_ct.jsonl"
    json.dump([cfg], open(W + "/cfg.json", "w"))
    def sh(cmd):
        print("RUN", cmd, flush=True); r = subprocess.run(cmd, shell=True, cwd=W, env=os.environ); print("rc", r.returncode, time.time() - t0, flush=True); return r.returncode
    sh("python dense_stream.py cfg.json")
    qd = f"{W}/hf/Qwen2.5-72B/q_{name}"; meta = json.load(open(qd + "/meta.json")) if os.path.exists(qd + "/meta.json") else None
    if meta is None: return dict(name=name, error="quant failed")
    gg = cfg["g"]
    sh(f"python export_hf.py {P} {qd} {W}/exp {gg} fp8 qcfg.json")
    sh(f"python vllm_base.py {W}/exp {W}/t1.json {W}/t1v2 bf16 1")
    sh(f"python vllm_ct.py {W}/exp {W}/ct.json items.jsonl")
    res = dict(name=name, cfg=cfg, meta=meta, secs=time.time() - t0)
    for k, f in (("t1", "t1.json"), ("ct", "ct.json")):
        if os.path.exists(W + "/" + f): res[k] = json.load(open(W + "/" + f))
    os.makedirs("/vol/72B", exist_ok=True); json.dump(res, open(f"/vol/72B/{name}.json", "w")); vol.commit()
    return res

if __name__ == "__main__":
    vs = json.load(open(sys.argv[1])); os.makedirs("modal_results", exist_ok=True); calls = []
    with modal.enable_output(), app.run(detach=True):
        for v in vs: calls.append((v["name"], run.spawn(v).object_id)); print(calls[-1], flush=True)
    json.dump(calls, open(sys.argv[1] + ".calls", "w"))
