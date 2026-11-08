"""modal run --detach modal_l7.py : L7 feedback experiment (A10G). Volume tenpct-ne /vol/l7/"""
import modal, os, subprocess, shutil
HERE = os.path.dirname(os.path.abspath(__file__))
app = modal.App("tenpct-l7-native")
img = (modal.Image.debian_slim(python_version="3.11").pip_install("torch==2.5.1", "transformers==4.57.1", "accelerate", "tokenizers", "safetensors", "huggingface_hub", "numpy").add_local_dir(HERE, "/root/l7", ignore=["__pycache__", "results"]))
vol = modal.Volume.from_name("tenpct-ne", create_if_missing=True)
@app.function(image=img, gpu="A10G", cpu=8, memory=32768, timeout=3 * 3600, volumes={"/vol": vol})
def run():
    env = dict(os.environ, HF_HOME="/vol/hf"); p = subprocess.run(["python", "run_l7.py"], cwd="/root/l7", env=env, capture_output=True, text=True)
    print(p.stdout[-8000:]); print(p.stderr[-4000:]); os.makedirs("/vol/l7", exist_ok=True)
    for f in os.listdir("/root/l7/results") if os.path.isdir("/root/l7/results") else []: shutil.copy("/root/l7/results/" + f, "/vol/l7/")
    open("/vol/l7/log.txt", "w").write(p.stdout[-20000:] + "\nSTDERR\n" + p.stderr[-8000:]); vol.commit()
@app.local_entrypoint()
def main(): run.spawn(); print("spawned")
