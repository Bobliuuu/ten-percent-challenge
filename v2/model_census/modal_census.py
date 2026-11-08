"""Model census launcher: python modal_census.py jobs_census.json (spawns detached). Results -> volume tenpct-tasksearch:/<tag>_cen.json (+dense/ct_*)."""
import modal, json, sys, os
HERE = os.path.dirname(os.path.abspath(__file__)); TS = HERE + "/../task_search"
app = modal.App("tenpct-census")
vol = modal.Volume.from_name("tenpct-tasksearch", create_if_missing=True)
hfvol = modal.Volume.from_name("tenpct-v2-hf", create_if_missing=True)
img = (modal.Image.debian_slim(python_version="3.11").pip_install("torch==2.4.1", "transformers==4.57.1", "numpy", "accelerate", "huggingface_hub", "sentencepiece", "protobuf")
       .add_local_file(TS + "/dense_core.py", "/code/dense_core.py").add_local_file(HERE + "/census_items.jsonl", "/code/census_items.jsonl")
       .add_local_file(HERE + "/census_calib.jsonl", "/code/census_calib.jsonl").add_local_file(TS + "/../compress/dense/dense_data.json", "/code/dense_data.json"))

def _go(job):
    import os, sys
    os.environ.update(HF_HOME="/hf", DENSE_HF="/vol", DENSE_DATA="/code/dense_data.json", CT_ITEMS="/code/census_items.jsonl", CT_CALIB="/code/census_calib.jsonl", HF_HUB_ENABLE_HF_TRANSFER="0")
    os.makedirs("/vol/dense", exist_ok=True)
    if job.get("multi"): os.environ["DENSE_MULTI"] = "1"
    if job.get("gpu_load", True): os.environ["DENSE_GPU"] = "1"
    sys.path.insert(0, "/code"); import dense_core
    cfgs = [dict(model=job["model"], ctasks=1, calib_ct=1, gen=True, **c) if not c["name"].endswith("bf16") else dict(model=job["model"], name=c["name"], mode="ctbase", gen=True) for c in job["cfgs"]]
    res = dense_core._run(dict(multi=cfgs)); tag = job["model"].split("/")[-1]; out = {}
    for c in cfgs:
        f = f"/vol/dense/ct_{tag}_{c['name']}.json"
        if os.path.exists(f): out[c["name"]] = json.load(open(f))
    json.dump(out, open(f"/vol/{tag}_{job['tag']}.json", "w")); vol.commit(); hfvol.commit()
    print("RESULT_DONE", tag, list(out), flush=True); return dict(errors=[r for r in res if isinstance(r, dict) and "error" in r], names=list(out))
SZ = dict(xl=dict(gpu="A100-80GB:4", memory=420000), s=dict(gpu="A100-80GB", memory=64000), m=dict(gpu=["H100", "A100-80GB"], memory=120000))
fns = {k: app.function(image=img, timeout=24 * 3600, volumes={"/vol": vol, "/hf": hfvol}, name="go_" + k, **v)(_go) for k, v in SZ.items()}
if __name__ == "__main__":
    jobs = json.load(open(sys.argv[1])); calls = []
    with modal.enable_output(), app.run(detach=True):
        for j in jobs: calls.append((j["model"], j["tag"], fns[j.get("size", "s")].spawn(j).object_id)); print(calls[-1], flush=True)
    json.dump(calls, open(sys.argv[1] + ".calls", "w"))
