"""L5 BANK6 exam run: dev-selected seed (d0), data incl. exam rows, scored once, packs to /r/pkb6_0. modal run --detach modal_b6f.py
  modal run --detach modal_b6.py"""
import modal, os
HERE = os.path.dirname(os.path.abspath(__file__))
app = modal.App("tenpct-l5-b6f")
img = (modal.Image.debian_slim(python_version="3.11").pip_install("torch==2.8.0", "transformers==4.57.1", "accelerate", "huggingface_hub", "numpy", "constriction", "safetensors")
       .add_local_dir(HERE + "/repb6f", "/root/rep"))
vol = modal.Volume.from_name("tenpct-l5"); hf = modal.Volume.from_name("tenpct-v2-hf")
@app.function(image=img, gpu="H200", cpu=16, memory=200000, timeout=43200, volumes={"/r": vol, "/hf": hf}, env={"HF_HOME": "/hf", "REP_DATA": "/root/rep/data", "REP_CALIB": "calib.jsonl"})
def run(name: str):
    import subprocess, sys, shutil
    p = subprocess.Popen([sys.executable, "run_arm.py", "cfg.json", name], cwd="/root/rep", stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in p.stdout: print(line, end="", flush=True)
    p.wait(); shutil.copytree("/root/out", f"/r/b6f/{name}", dirs_exist_ok=True); vol.commit(); print("EXIT", p.returncode, flush=True)
@app.local_entrypoint()
def main():
    hs = [run.spawn(f"B6HF_d0") for s in range(1)]
    for h in hs: h.get()
