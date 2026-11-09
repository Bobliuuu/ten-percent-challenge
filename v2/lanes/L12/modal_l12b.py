"""Modal app tenpct-l12b: L12 feedback jobs (o_proj run_fbo.py, shared-expert carrier run_fbs.py) and the packer job. Deploy then spawn via launch_l12b.py. Results on volume tenpct-l12b:/r/<tag> and printed (FBO/FBS lines)."""
import modal, os
HERE = os.path.dirname(os.path.abspath(__file__)); V2 = os.path.abspath(HERE + "/../..")
app = modal.App("tenpct-l12b")
img = (modal.Image.debian_slim(python_version="3.11").pip_install("torch==2.10.0", "transformers==5.22.2", "safetensors", "scipy", "huggingface_hub", "numpy", "accelerate")
       .add_local_dir(HERE, "/workspace/fin/v2/lanes/L12", ignore=["__pycache__", "*.pyc", "results", "data_bank"])
       .add_local_dir(V2 + "/exam", "/workspace/fin/v2/exam", ignore=["__pycache__"])
       .add_local_dir(V2 + "/task_search/EASY", "/workspace/fin/v2/task_search/EASY").add_local_dir(V2 + "/moe_quant/data", "/workspace/fin/v2/moe_quant/data"))
vol = modal.Volume.from_name("tenpct-l12b", create_if_missing=True); hf = modal.Volume.from_name("tenpct-v2-hf")
@app.function(image=img, gpu="A100-80GB", cpu=8, memory=230000, timeout=86400, ephemeral_disk=600000, volumes={"/r": vol, "/hf": hf}, env={"HF_HOME": "/hf"})
def run(tag, script, args, files=None):
    import subprocess, sys, json
    L = "/workspace/fin/v2/lanes/L12"; os.makedirs(f"/r/{tag}", exist_ok=True)
    for k, v in (files or {}).items(): open(f"{L}/{k}", "w").write(json.dumps(v))
    cmd = [sys.executable, f"{L}/{script}"] + args + ["--out", f"/r/{tag}", "--root", "/workspace/fin"]
    p = subprocess.Popen(cmd, cwd=L, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in p.stdout:
        print(line, end="", flush=True)
        if any(x in line for x in ("FBO ", "FBS ", "EMPTY", "DONE", "BYTES", "PACK")): vol.commit()
    p.wait(); vol.commit()
    print("EXIT", p.returncode)
