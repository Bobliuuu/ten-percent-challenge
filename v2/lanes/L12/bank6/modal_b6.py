"""Modal app tenpct-l12-b6 (L12 on BANK6). Deploy, then spawn via launch_b6.py. Volume tenpct-l12-b6:/r/<tag>."""
import modal, os
HERE = os.path.dirname(os.path.abspath(__file__)); L12 = os.path.dirname(HERE); V2 = os.path.dirname(os.path.dirname(L12))
app = modal.App("tenpct-l12-b6")
img = (modal.Image.debian_slim(python_version="3.11").pip_install("torch==2.10.0", "transformers==5.22.2", "safetensors", "scipy", "huggingface_hub", "numpy", "accelerate")
       .add_local_file(L12 + "/l12_core.py", "/workspace/fin/v2/lanes/L12/l12_core.py").add_local_file(L12 + "/l12_fb.py", "/workspace/fin/v2/lanes/L12/l12_fb.py")
       .add_local_dir(HERE, "/workspace/fin/v2/lanes/L12/bank6", ignore=["__pycache__", "*.pyc", "results"])
       .add_local_dir(V2 + "/exam", "/workspace/fin/v2/exam", ignore=["__pycache__"])
       .add_local_dir(V2 + "/task_search/EASY", "/workspace/fin/v2/task_search/EASY").add_local_dir(V2 + "/moe_quant/data", "/workspace/fin/v2/moe_quant/data"))
vol = modal.Volume.from_name("tenpct-l12-b6", create_if_missing=True); hf = modal.Volume.from_name("tenpct-v2-hf")
@app.function(image=img, gpu="A100-80GB", cpu=8, memory=230000, timeout=86400, ephemeral_disk=600000, volumes={"/r": vol, "/hf": hf}, env={"HF_HOME": "/hf"})
def run(tag, args, files=None):
    import subprocess, sys, json
    L = "/workspace/fin/v2/lanes/L12/bank6"; os.makedirs(f"/r/{tag}", exist_ok=True)
    for k, v in (files or {}).items(): open(f"{L}/{k}", "w").write(json.dumps(v))
    cmd = [sys.executable, f"{L}/run_b6.py"] + args + ["--out", f"/r/{tag}", "--root", "/workspace/fin"]
    p = subprocess.Popen(cmd, cwd=L, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in p.stdout:
        print(line, end="", flush=True)
        if any(x in line for x in ("FBDEV ", "FBLOCK ", "EMPTY", "DONE", "EXAM", "RESULT")): vol.commit()
    p.wait(); vol.commit(); print("EXIT", p.returncode)
