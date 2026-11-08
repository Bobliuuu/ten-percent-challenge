"""L10 Modal launcher. python3 -m modal run --detach is NOT used: deploy-less spawn pattern as v2/task_search/modal_run.py.
python modal_l10.py jobs.json -> spawns detached; results in Volume tenpct-tasksearch:/dense/ct_l10_text_<name>.json and printed RESULT_DONE."""
import modal, json, sys, os
HERE = os.path.dirname(os.path.abspath(__file__))
app = modal.App("tenpct-l10")
vol = modal.Volume.from_name("tenpct-tasksearch", create_if_missing=True)
hfvol = modal.Volume.from_name("tenpct-v2-hf", create_if_missing=True)
img = (modal.Image.debian_slim(python_version="3.11").pip_install("torch==2.4.1", "transformers==4.55.0", "numpy", "accelerate", "huggingface_hub", "safetensors")
       .add_local_file(HERE + "/dense_core_l10.py", "/code/dense_core.py").add_local_file(HERE + "/prep_text.py", "/code/prep_text.py")
       .add_local_file(HERE + "/data/items_dev.jsonl", "/code/easy_items.jsonl").add_local_file(HERE + "/data/calib.jsonl", "/code/easy_calib.jsonl")
       .add_local_file(HERE + "/../../compress/dense/dense_data.json", "/code/dense_data.json"))

def _go(job):
    import os, sys
    os.environ.update(HF_HOME="/hf", DENSE_HF="/vol", DENSE_DATA="/code/dense_data.json", CT_ITEMS="/code/easy_items.jsonl", CT_CALIB="/code/easy_calib.jsonl", HF_HUB_ENABLE_HF_TRANSFER="0", DENSE_GPU="1")
    os.makedirs("/vol/dense", exist_ok=True)
    sys.path.insert(0, "/code")
    import prep_text; M = prep_text.ensure(job["src"], "/hf/" + job["dirname"], job.get("verify", False), job.get("kind", "mllama")); hfvol.commit()
    import dense_core
    cfgs = []
    for c in job["cfgs"]:
        if c["name"].endswith("bf16"): cfgs.append(dict(model=M, name=c["name"], mode="ctbase", gen=True, gen_bs=c.get("gen_bs", 48), ct_bs=8))
        else: cfgs.append(dict(model=M, ctasks=1, calib_ct=1, gen=True, base_bytes_override=job["base_bytes"], **c))
    res = dense_core._run(dict(multi=cfgs)); tag = os.path.basename(M); out = {}
    for c in cfgs:
        f = f"/vol/dense/ct_{tag}_{c['name']}.json"
        if os.path.exists(f): out[c["name"]] = json.load(open(f))
    json.dump(out, open(f"/vol/{tag}_{job['tag']}.json", "w")); vol.commit(); hfvol.commit()
    print("RESULT_DONE", tag, job["tag"], list(out), flush=True); return dict(errors=[r for r in res if isinstance(r, dict) and "error" in r], names=list(out))
fn = app.function(image=img, timeout=8 * 3600, volumes={"/vol": vol, "/hf": hfvol}, gpu="A100-80GB", memory=96000, name="go_l10")(_go)
if __name__ == "__main__":
    jobs = json.load(open(sys.argv[1])); calls = []
    with modal.enable_output(), app.run(detach=True):
        for j in jobs: calls.append((j["tag"], fn.spawn(j).object_id)); print(calls[-1], flush=True)
    json.dump(calls, open(sys.argv[1] + ".calls", "w"))
