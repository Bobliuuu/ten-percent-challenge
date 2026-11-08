"""Modal runner for ne_probe.py (lane e). modal run --detach modal_ne.py --models "microsoft/bitnet-b1.58-2B-4T:bitnet2b,tiiuae/Falcon-E-3B-Instruct:falcone3b"
Results -> Volume tenpct-ne (/vol/<tag>/) and printed RESULT_JSON lines."""
import modal, os, subprocess, shutil, json
HERE = os.path.dirname(os.path.abspath(__file__))
app = modal.App("tenpct-near-miss-e")
img = (modal.Image.debian_slim(python_version="3.11").pip_install("torch==2.5.1", "transformers==4.57.1", "accelerate", "tokenizers", "safetensors", "huggingface_hub", "sentencepiece", "numpy")
       .add_local_dir(HERE + "/bundle", "/root/bundle"))
vol = modal.Volume.from_name("tenpct-ne", create_if_missing=True)
@app.function(image=img, gpu="A10G", cpu=8, memory=49152, timeout=4 * 3600, volumes={"/vol": vol})
def run(model, tag):
    env = dict(os.environ, HF_HOME="/vol/hf")
    os.makedirs("/root/bundle/results", exist_ok=True)
    p = subprocess.run(["python", "code/ne_probe.py", model, tag, "--budget", "0.0985"], cwd="/root/bundle", env=env, capture_output=True, text=True)
    print(p.stdout[-6000:]); print(p.stderr[-3000:])
    dst = f"/vol/{tag}"; os.makedirs(dst, exist_ok=True)
    for f in os.listdir("/root/bundle/results"): shutil.copy("/root/bundle/results/" + f, dst)
    vol.commit()
    return tag
@app.local_entrypoint()
def main(models: str):
    hs = [run.spawn(*m.split(":")) for m in models.split(",")]
    for h in hs: print("DONE", h.get())
