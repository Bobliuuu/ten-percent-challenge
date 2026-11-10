"""Modal app tenpct-l12-owndata. Deploy then spawn: modal deploy modal_owndata.py ; python launch_owndata.py <tag>. Results on volume tenpct-l12b:/r/<tag> and printed as RESULT_JSON."""
import modal, os
HERE = os.path.dirname(os.path.abspath(__file__)); L12 = os.path.abspath(HERE + "/.."); V2 = os.path.abspath(HERE + "/../../..")
app = modal.App("tenpct-l12-owndata")
img = (modal.Image.debian_slim(python_version="3.11").pip_install("torch==2.10.0", "transformers==5.22.2", "safetensors", "scipy", "huggingface_hub", "numpy", "accelerate")
       .add_local_dir(L12, "/workspace/fin/v2/lanes/L12", ignore=["__pycache__", "*.pyc", "results", "data_bank"])
       .add_local_dir(V2 + "/exam", "/workspace/fin/v2/exam", ignore=["__pycache__"])
       .add_local_dir(V2 + "/task_search/EASY", "/workspace/fin/v2/task_search/EASY").add_local_dir(V2 + "/moe_quant/data", "/workspace/fin/v2/moe_quant/data"))
vol = modal.Volume.from_name("tenpct-l12b", create_if_missing=True); hf = modal.Volume.from_name("tenpct-v2-hf")
@app.function(image=img, gpu="A100-80GB", cpu=8, memory=230000, timeout=86400, ephemeral_disk=600000, volumes={"/r": vol, "/hf": hf}, env={"HF_HOME": "/hf"})
def run(tag):
    import subprocess, sys
    L = "/workspace/fin/v2/lanes/L12"; os.makedirs(f"/r/{tag}", exist_ok=True)
    p = subprocess.Popen([sys.executable, f"{L}/owndata/owndata_job.py", "--out", f"/r/{tag}", "--root", "/workspace/fin"], cwd=L + "/owndata", stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in p.stdout:
        print(line, end="", flush=True)
        if any(x in line for x in ("NOFB", "RECIPE25", "timer ", "OD_DONE", "base done")): vol.commit()
    p.wait(); vol.commit(); print("EXIT", p.returncode)
