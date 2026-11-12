"""Modal app for the deletion-feedback lane (own app so the shipped tenpct-l9b deployment is untouched). Deploy: modal deploy modal_del.py ; launch: python3 launch_del.py job.json"""
import modal, os, sys
app = modal.App("tenpct-l9b-del")
HERE = os.path.dirname(os.path.abspath(__file__)); L9 = os.path.dirname(HERE)
img = (modal.Image.debian_slim(python_version="3.11").pip_install("torch==2.8.0", "transformers==5.22.2", "numpy", "accelerate", "huggingface_hub", "sentencepiece", "protobuf", "tiktoken", "safetensors", "kernels")
       .add_local_dir(L9, "/root/l9b", ignore=["__pycache__", "*.pyc", "bank6q", "results", "deletion/results"]))
vol = modal.Volume.from_name("tenpct-l9b")
hfvol = modal.Volume.from_name("tenpct-v2-hf", create_if_missing=True)
sys.path.insert(0, "/root/l9b"); sys.path.insert(0, L9)
import runjob
def main(job):
    runjob.COMMIT = lambda: vol.commit(); return runjob.main(job)
f = app.function(image=img, timeout=24 * 3600, volumes={"/r": vol, "/hf": hfvol}, env={"HF_HOME": "/hf", "HF_HUB_ENABLE_HF_TRANSFER": "0"}, name="run_s", gpu="A100-80GB", memory=96000, scaledown_window=2)(main)
