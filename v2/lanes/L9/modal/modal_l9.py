"""L9 Modal app: Qwen3-235B-A22B-Instruct-2507 packed artifact. `modal deploy modal_l9.py`; launch with launch_l9.py (spawn, results on the private volume + RESULT_JSON in logs).
Volumes (private, workspace-only): tenpct-l9-hf (HF snapshot, 470 GB), tenpct-l9-art (packed artifacts + results). No ids/IPs are written into the repo."""
import modal, os, subprocess, sys, json, time
app = modal.App("tenpct-l9")
HERE = os.path.dirname(os.path.abspath(__file__)); L9 = os.path.dirname(HERE); V2 = os.path.dirname(os.path.dirname(L9))
img = (modal.Image.debian_slim(python_version="3.11").pip_install("torch==2.5.1", "transformers==4.56.2", "numpy", "accelerate", "huggingface_hub", "hf_transfer", "sentencepiece", "protobuf", "safetensors", "tiktoken")
       .add_local_dir(L9, "/root/l9", ignore=["__pycache__", "*.pyc", "data/exam_skill7.jsonl", "data/bank8_*"])
       .add_local_dir(V2 + "/exam", "/root/v2/exam", ignore=["__pycache__"])
       .add_local_dir(V2 + "/task_search/EASY", "/root/v2/task_search/EASY")
       .add_local_dir(V2 + "/moe_quant/data", "/root/v2/moe_quant/data")
       .add_local_file(V2 + "/tools/bytes_audit.py", "/root/v2/tools/bytes_audit.py"))
hf = modal.Volume.from_name("tenpct-l9-hf", create_if_missing=True, version=2)
art = modal.Volume.from_name("tenpct-l9-art", create_if_missing=True, version=2)
VOLS = {"/hf": hf, "/art": art}
ENV = {"HF_HOME": "/hf/home", "HF_HUB_ENABLE_HF_TRANSFER": "1", "PYTHONUNBUFFERED": "1", "L9_GPUMEM": "70GiB", "PYTHONPATH": "/root/l9"}

def _sh(cmd, cwd="/root/l9"):
    print("RUN", cmd, flush=True); t = time.time()
    r = subprocess.run(cmd, shell=True, cwd=cwd, env={**os.environ, "PYTHONPATH": "/root/l9"}); art.commit(); hf.commit()
    print("EXIT", r.returncode, round(time.time() - t), flush=True); return r.returncode

@app.function(image=img, cpu=16, memory=65536, timeout=12 * 3600, volumes=VOLS, env=ENV)
def dl():
    from huggingface_hub import snapshot_download
    p = snapshot_download("Qwen/Qwen3-235B-A22B-Instruct-2507", local_dir="/hf/qwen235", max_workers=32, allow_patterns=["*.json", "*.safetensors", "*.txt"])
    hf.commit(); print("RESULT_JSON", json.dumps(dict(dl=p, files=len(os.listdir(p))))); return p

@app.function(image=img, gpu="H200", cpu=8, memory=65536, timeout=6 * 3600, volumes=VOLS, env=ENV)
def g1(cmd): return _sh(cmd)
@app.function(image=img, gpu="H200:4", cpu=32, memory=327680, timeout=6 * 3600, volumes=VOLS, env=ENV)
def g4(cmd): return _sh(cmd)
@app.function(image=img, gpu="H200:5", cpu=32, memory=327680, timeout=6 * 3600, volumes=VOLS, env=ENV)
def g5(cmd): return _sh(cmd)
@app.function(image=img, cpu=8, memory=32768, timeout=6 * 3600, volumes=VOLS, env=ENV)
def cpu(cmd): return _sh(cmd)

@app.function(image=img, gpu="H100:8", cpu=32, memory=327680, timeout=8 * 3600, volumes=VOLS, env=ENV)
def g8h(cmd): return _sh(cmd)

@app.function(image=img, gpu="H200:3", cpu=32, memory=327680, timeout=8 * 3600, volumes=VOLS, env=ENV)
def g3(cmd): return _sh(cmd)
@app.function(image=img, gpu="A100-80GB:6", cpu=32, memory=327680, timeout=8 * 3600, volumes=VOLS, env=ENV)
def a6(cmd): return _sh(cmd)
