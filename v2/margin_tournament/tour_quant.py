"""Empirical check: quantise with the lane's 1.57 bpw (10.0% target_ratio) recipe at 2 random draws (rotation signs + calibration order),
evaluate on the task's dev items. Sequential, 1 x A100-80GB. python3 tour_quant.py <items> <calib> <model> <task1,task2,..>"""
import modal, json, sys, os
HERE = os.path.dirname(os.path.abspath(__file__))
app = modal.App("tenpct-margintour-q"); vol = modal.Volume.from_name("tenpct-margintour", create_if_missing=True); hfvol = modal.Volume.from_name("tenpct-v2-hf", create_if_missing=True)
A = sys.argv[1:]
img = (modal.Image.debian_slim(python_version="3.11").pip_install("torch==2.4.1", "transformers==4.55.0", "numpy", "accelerate", "huggingface_hub")
       .add_local_file(HERE + "/dense_core_tour.py", "/code/dense_core.py").add_local_file(HERE + "/../compress/dense/dense_data.json", "/code/dense_data.json")
       .add_local_file(A[0] if A else HERE + "/x", "/code/items.jsonl").add_local_file(A[1] if A else HERE + "/x", "/code/calib.jsonl"))
@app.function(image=img, gpu="A100-80GB", memory=64000, timeout=8 * 3600, volumes={"/vol": vol, "/hf": hfvol})
def run(model, tasks):
    import sys, os, json
    os.environ.update(HF_HOME="/hf", DENSE_HF="/vol", DENSE_DATA="/code/dense_data.json", HF_HUB_ENABLE_HF_TRANSFER="0", DENSE_GPU="1"); os.makedirs("/vol/dense", exist_ok=True)
    sys.path.insert(0, "/code"); import dense_core
    for t in tasks:
        for f in ("items", "calib"):
            open(f"/tmp/{f}_{t}.jsonl", "w").writelines(l for l in open(f"/code/{f}.jsonl") if json.loads(l)["task"] == t)
        os.environ.update(CT_ITEMS=f"/tmp/items_{t}.jsonl", CT_CALIB=f"/tmp/calib_{t}.jsonl")
        for dr in (0, 1):
            cfg = dict(model=model, ctasks=1, calib_ct=1, name=f"{t}_d{dr}", target_ratio=0.1, emb_L=4, head_L=4, g=256, sign_seed=dr, calib_seed=dr)
            try: dense_core._run(cfg)
            except Exception as e:
                import traceback; traceback.print_exc()
            vol.commit(); print("Q_DONE", t, dr, flush=True)
if __name__ == "__main__":
    with modal.enable_output(), app.run(detach=True):
        c = run.spawn(A[2], A[3].split(",")); print("CALL", c.object_id, flush=True)
