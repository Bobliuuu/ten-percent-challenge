"""Locked SKILL7 exam for the shipped default `scale` carrier (run_locked_scale.py) on a Modal GPU with the PRIVATE volume artifact (tenpct-l1-art:/art). Deploy + spawn; result JSON written to volume tenpct-l1-art:/results/locked_L1_scale.json and printed as RESULT_JSON.
  modal deploy v2/lanes/L1/modal_locked_scale.py ; python v2/lanes/L1/modal_locked_scale.py spawn"""
import modal, os, sys
R = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
app = modal.App("tenpct-l1-lockscale"); vol = modal.Volume.from_name("tenpct-l1-art")
ig = ["**/__pycache__", "**/results/**", "**/build/**"]
img = (modal.Image.debian_slim(python_version="3.11").pip_install("torch==2.5.1", "transformers==4.56.2", "numpy", "accelerate", "huggingface_hub")
       .add_local_dir(R + "/v2/lanes/L1", "/repo/v2/lanes/L1", ignore=ig).add_local_dir(R + "/v2/exam", "/repo/v2/exam", ignore=["**/__pycache__"])
       .add_local_dir(R + "/v2/task_search/EASY", "/repo/v2/task_search/EASY").add_local_dir(R + "/v2/moe_quant/data", "/repo/v2/moe_quant/data"))
@app.function(image=img, gpu="L40S", timeout=6 * 3600, volumes={"/vol": vol})
def run():
    import subprocess, shutil, os, json
    shutil.copytree("/vol/art", "/work/art"); L = "/repo/v2/lanes/L1"
    for f in ("answer.py", "feedback.py", "bonsai_loader.py", "common.py", "skill_task.json", "feedback_calib.jsonl", "tensor_manifest.json"): shutil.copy(L + "/" + f, "/work/art/" + f)
    from huggingface_hub import snapshot_download
    snap = snapshot_download("prism-ml/Bonsai-8B-unpacked", revision="376f381570d6115bc03f82adcfa4af0c7672ae54")
    p = subprocess.run(["python", "-u", L + "/run_locked_scale.py", "/work/art", snap, "/repo", "/work/out.json"], stdout=open("/work/log.txt", "w"), stderr=subprocess.STDOUT)
    os.makedirs("/vol/results", exist_ok=True); shutil.copy("/work/log.txt", "/vol/results/locked_L1_scale.log")
    if os.path.exists("/work/out.json"): shutil.copy("/work/out.json", "/vol/results/locked_L1_scale.json")
    vol.commit(); print("RESULT_JSON", open("/work/out.json").read() if os.path.exists("/work/out.json") else "FAILED", flush=True); print(open("/work/log.txt").read()[-2000:])
if __name__ == "__main__":
    f = modal.Function.from_name("tenpct-l1-lockscale", "run"); c = f.spawn(); print("CALL", c.object_id)
