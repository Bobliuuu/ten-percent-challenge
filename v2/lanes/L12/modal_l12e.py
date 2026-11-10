"""Modal app tenpct-l12e: L12 end-to-end from the PACKED files (prep/parity/cli/locked jobs). Deploy, then spawn via launch_l12e.py. Results on volume tenpct-l12b:/e2e/<tag> and printed (E2E lines).
Artifact dir (private volume, never public): tenpct-l12b:/art ."""
import modal, os
HERE = os.path.dirname(os.path.abspath(__file__)); V2 = os.path.abspath(HERE + "/../..")
app = modal.App("tenpct-l12e")
img = (modal.Image.debian_slim(python_version="3.11").pip_install("torch==2.10.0", "transformers==5.22.2", "safetensors", "scipy", "huggingface_hub", "numpy", "accelerate")
       .add_local_dir(HERE, "/workspace/fin/v2/lanes/L12", ignore=["__pycache__", "*.pyc", "results", "data_bank", "devsel", "owndata", "bank6"])
       .add_local_dir(V2 + "/exam", "/workspace/fin/v2/exam", ignore=["__pycache__"])
       .add_local_dir(V2 + "/task_search/EASY", "/workspace/fin/v2/task_search/EASY").add_local_dir(V2 + "/moe_quant/data", "/workspace/fin/v2/moe_quant/data"))
vol = modal.Volume.from_name("tenpct-l12b", create_if_missing=True); hf = modal.Volume.from_name("tenpct-v2-hf")
def _run(tag, script, args):
    import subprocess, sys
    L = "/workspace/fin/v2/lanes/L12"; os.makedirs(f"/r/e2e/{tag}", exist_ok=True)
    cmd = [sys.executable, f"{L}/{script}"] + args + ["--out", f"/r/e2e/{tag}", "--root", "/workspace/fin"]
    p = subprocess.Popen(cmd, cwd=L, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in p.stdout:
        print(line, end="", flush=True)
        if "E2E " in line: vol.commit()
    p.wait(); vol.commit(); print("EXIT", p.returncode)
@app.function(image=img, cpu=4, memory=32000, timeout=7200, ephemeral_disk=524288, volumes={"/r": vol, "/hf": hf}, env={"HF_HOME": "/hf"})
def prep(tag, script, args): _run(tag, script, args)
@app.function(image=img, gpu="A100-80GB", cpu=8, memory=230000, timeout=86400, ephemeral_disk=600000, volumes={"/r": vol, "/hf": hf}, env={"HF_HOME": "/hf"})
def run(tag, script, args): _run(tag, script, args)
