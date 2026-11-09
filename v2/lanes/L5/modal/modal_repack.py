"""Rebuild the L5 packed body (S7F_d0 config, seed 0) on Modal and keep it in the PRIVATE volume tenpct-l5 (/r/pk = packed artifact, /r/q32_s7 = dequantised HF checkpoint used ONLY to verify decode).
  modal run --detach v2/lanes/L5/modal/modal_repack.py"""
import modal, os
HERE = os.path.dirname(os.path.abspath(__file__))
app = modal.App("tenpct-l5-repack")
img = (modal.Image.debian_slim(python_version="3.11").pip_install("torch==2.8.0", "transformers==4.57.1", "accelerate", "huggingface_hub", "numpy", "constriction", "safetensors")
       .add_local_dir(HERE + "/rep", "/root/rep"))
vol = modal.Volume.from_name("tenpct-l5", create_if_missing=True); hf = modal.Volume.from_name("tenpct-v2-hf")
@app.function(image=img, gpu="H200", cpu=16, memory=200000, timeout=43200, volumes={"/r": vol, "/hf": hf}, env={"HF_HOME": "/hf", "REP_DATA": "/root/rep/data", "REP_CALIB": "calib.jsonl"})
def run():
    import subprocess, sys
    p = subprocess.Popen([sys.executable, "run_arm.py", "cfg.json", "S7F_d0"], cwd="/root/rep", stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in p.stdout: print(line, end="", flush=True)
    p.wait(); vol.commit(); print("EXIT", p.returncode, flush=True)
@app.local_entrypoint()
def main(): run.remote()
