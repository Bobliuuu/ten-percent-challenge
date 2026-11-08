"""Modal app tenpct-l12: runs l12_core on A100-80GB (deploy, then spawn via launch_modal.py). Results: volume tenpct-l12 at /r/<tag>, also printed."""
import modal, os
HERE = os.path.dirname(os.path.abspath(__file__))
app = modal.App("tenpct-l12")
img = (modal.Image.debian_slim(python_version="3.11").pip_install("torch==2.10.0", "transformers==5.22.2", "safetensors", "scipy", "huggingface_hub", "numpy", "accelerate")
       .add_local_file(HERE + "/l12_core.py", "/root/l12_core.py").add_local_file(HERE + "/l12_fb.py", "/root/l12_fb.py").add_local_file(HERE + "/run_fb.py", "/root/run_fb.py")
       .add_local_dir(HERE + "/data", "/root/data"))
vol = modal.Volume.from_name("tenpct-l12", create_if_missing=True); hf = modal.Volume.from_name("tenpct-v2-hf")
@app.function(image=img, gpu="A100-80GB", cpu=8, memory=230000, timeout=86400, volumes={"/r": vol, "/hf": hf}, env={"HF_HOME": "/hf"})
def run(tag, cfgs, extra=None):
    import subprocess, json, sys
    os.makedirs(f"/r/{tag}", exist_ok=True); json.dump(cfgs, open(f"/r/{tag}/cfgs.json", "w"))
    cmd = [sys.executable, "l12_core.py", "--cfgs", f"/r/{tag}/cfgs.json", "--data", "data", "--out", f"/r/{tag}"] if not extra else extra
    p = subprocess.Popen(cmd, cwd="/root", stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in p.stdout:
        print(line, end="", flush=True)
        if "RESULT" in line: vol.commit()
    p.wait(); vol.commit()
