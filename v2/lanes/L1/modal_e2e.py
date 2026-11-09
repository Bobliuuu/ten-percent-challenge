"""Shipped-CLI protocol on a Modal GPU with the PRIVATE volume artifact (tenpct-l1-art:/art): e2e_locked.py (feedback.py -> patch.pt -> answer.py; arms real/shuffled/wrong/empty) with the shipped DEFAULT carrier.
  modal run --detach v2/lanes/L1/modal_e2e.py --tag scale_default"""
import modal, os
R = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
app = modal.App("tenpct-l1-e2e"); vol = modal.Volume.from_name("tenpct-l1-art")
ig = ["**/__pycache__", "**/results/**", "**/build/**"]
img = (modal.Image.debian_slim(python_version="3.11").pip_install("torch==2.5.1", "transformers==4.56.2", "numpy", "accelerate")
       .add_local_dir(R + "/v2/lanes/L1", "/repo/v2/lanes/L1", ignore=ig).add_local_dir(R + "/v2/exam", "/repo/v2/exam", ignore=["**/__pycache__"])
       .add_local_dir(R + "/v2/task_search/EASY", "/repo/v2/task_search/EASY").add_local_dir(R + "/v2/moe_quant/data", "/repo/v2/moe_quant/data"))
@app.function(image=img, gpu="L40S", timeout=4 * 3600, volumes={"/vol": vol})
def run(tag: str, extra: str = ""):
    import subprocess, shutil, json, os
    shutil.copytree("/vol/art", "/work/art"); L = "/repo/v2/lanes/L1"
    for f in ("answer.py", "feedback.py", "bonsai_loader.py", "common.py", "skill_task.json", "feedback_calib.jsonl", "tensor_manifest.json"): shutil.copy(L + "/" + f, "/work/art/" + f)
    os.makedirs("/work/w", exist_ok=True)
    p = subprocess.run(["python", L + "/e2e_locked.py", "/work/art", "/work/w", "/repo", "/work/out.json"], capture_output=True, text=True)
    print(p.stdout[-3000:], p.stderr[-2000:], flush=True)
    return open("/work/out.json").read() if os.path.exists("/work/out.json") else "FAILED\n" + p.stderr[-3000:]
@app.local_entrypoint()
def main(tag: str = "scale_default"):
    r = run.remote(tag); open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", f"e2e_locked_{tag}.json"), "w").write(r); print(r[:3000])
