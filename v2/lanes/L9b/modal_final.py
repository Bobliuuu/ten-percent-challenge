"""L9b final checks. Usage (detached): FRESH_CLONE=<dir of a fresh git clone's v2/lanes/L9b> COMMIT=<sha> modal run --detach modal_final.py::fresh   |   modal run --detach modal_final.py::owndata
Results: RESULT_JSON in app logs + /r/l9b/res on private volume tenpct-l9b (pulled into results/ afterwards)."""
import modal, os, json
app = modal.App("tenpct-l9b-final")
HERE = os.path.dirname(os.path.abspath(__file__)); FC = os.environ.get("FRESH_CLONE", HERE)
base = modal.Image.debian_slim(python_version="3.11").pip_install("torch==2.8.0", "transformers==5.22.2", "numpy", "accelerate", "huggingface_hub", "sentencepiece", "protobuf", "tiktoken", "safetensors", "kernels")
img_fresh = base.add_local_dir(FC, "/root/fresh", ignore=["__pycache__", "*.pyc", "bank6q", "results"])
img_own = base.add_local_dir(HERE, "/root/l9b", ignore=["__pycache__", "*.pyc", "bank6q", "results"]).add_local_file(os.path.join(HERE, "final_jobs.py"), "/root/l9b/final_jobs.py")
vol = modal.Volume.from_name("tenpct-l9b"); hfvol = modal.Volume.from_name("tenpct-v2-hf", create_if_missing=True)
@app.function(image=img_fresh, gpu="A100-80GB", memory=96000, timeout=4 * 3600, volumes={"/r": vol})
def fresh(commit: str = ""):
    import sys; sys.path.insert(0, "/root/fresh"); import final_jobs; r = final_jobs.fresh_clone("/root/fresh", commit); vol.commit(); return r
@app.function(image=img_own, gpu="A100-80GB:2", memory=256000, timeout=8 * 3600, volumes={"/r": vol, "/hf": hfvol}, env={"HF_HOME": "/hf", "HF_HUB_ENABLE_HF_TRANSFER": "0"})
def owndata():
    import sys; sys.path.insert(0, "/root/l9b"); import final_jobs; r = final_jobs.own_data(); vol.commit(); return r
