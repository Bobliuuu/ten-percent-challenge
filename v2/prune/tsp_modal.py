"""TASK_SEARCH Modal launcher. python modal_run.py jobs.json  (spawns detached; results -> Modal volume tenpct-tasksearch:/ct_*.json + printed calls file)
Each job = {model, size, cfgs:[...dense_core cfgs...]}. Runs dense_core._run (GPTQ+Hadamard+ECSQ, label-task teacher-forced eval)."""
import modal, json, sys, os
HERE = os.path.dirname(os.path.abspath(__file__))
app = modal.App("tenpct-prune")
vol = modal.Volume.from_name("tenpct-tasksearch", create_if_missing=True)
hfvol = modal.Volume.from_name("tenpct-v2-hf", create_if_missing=True)
img = (modal.Image.debian_slim(python_version="3.11").pip_install("torch==2.4.1", "transformers==4.55.0", "numpy", "accelerate", "huggingface_hub")
       .add_local_file(HERE + "/tsp_core.py", "/code/dense_core.py").add_local_file(HERE + "/../task_search/items2.jsonl", "/code/items2.jsonl")
       .add_local_file(HERE + "/../task_search/calib2.jsonl", "/code/calib2.jsonl").add_local_file(HERE + "/../task_search/NEWS/items_dev.jsonl", "/code/news_items.jsonl").add_local_file(HERE + "/../task_search/NEWS/calib.jsonl", "/code/news_calib.jsonl").add_local_file(HERE + "/../task_search/INJR/items_dev.jsonl", "/code/injr_items.jsonl").add_local_file(HERE + "/../task_search/INJR/calib.jsonl", "/code/injr_calib.jsonl").add_local_file(HERE + "/../compress/dense/dense_data.json", "/code/dense_data.json"))

def _go(job):
    import os, sys, shutil
    os.environ.update(HF_HOME="/hf", DENSE_HF="/vol", DENSE_DATA="/code/dense_data.json", CT_ITEMS="/code/items2.jsonl", CT_CALIB="/code/calib2.jsonl", HF_HUB_ENABLE_HF_TRANSFER="0")
    os.makedirs("/vol/dense", exist_ok=True)
    if job.get("injr"): os.environ.update(CT_ITEMS="/code/injr_items.jsonl", CT_CALIB="/code/injr_calib.jsonl")
    if job.get("news"): os.environ.update(CT_ITEMS="/code/news_items.jsonl", CT_CALIB="/code/news_calib.jsonl")
    if job.get("gpu_load", True): os.environ["DENSE_GPU"] = "1"
    sys.path.insert(0, "/code"); import dense_core
    cfgs = [dict(model=job["model"], ctasks=1, calib_ct=1, **c) if not c["name"].endswith("bf16") else dict(model=job["model"], name=c["name"], mode="ctbase") for c in job["cfgs"]]
    res = dense_core._run(dict(multi=cfgs)); tag = job["model"].split("/")[-1]; out = {}
    for c in cfgs:
        f = f"/vol/dense/ct_{tag}_{c['name']}.json"
        if os.path.exists(f): out[c["name"]] = json.load(open(f))
    json.dump(out, open(f"/vol/{tag}_{job['tag']}.json", "w")); vol.commit(); hfvol.commit()
    print("RESULT_DONE", tag, list(out), flush=True); return dict(errors=[r for r in res if isinstance(r, dict) and "error" in r], names=list(out))
SZ = dict(s=dict(gpu="A100-80GB", memory=64000), m=dict(gpu=["H100", "A100-80GB"], memory=120000))
fns = {k: app.function(image=img, timeout=6 * 3600, volumes={"/vol": vol, "/hf": hfvol}, name="go_" + k, **v)(_go) for k, v in SZ.items()}
if __name__ == "__main__":
    jobs = json.load(open(sys.argv[1])); calls = []
    with modal.enable_output(), app.run(detach=True):
        for j in jobs: calls.append((j["model"], j["tag"], fns[j.get("size", "s")].spawn(j).object_id)); print(calls[-1], flush=True)
    json.dump(calls, open(sys.argv[1] + ".calls", "w"))
