"""L9b Modal app (Qwen3.5-35B-A3B real packed artifact). Deploy: modal deploy modal_l9b.py ; launch: python3 launch_l9b.py job.json  (spawn; results on volume tenpct-l9b:/res + RESULT_JSON in logs).
Stages: pack (2 GPUs: base outputs, routing-mass chunks, write packed artifacts), eval (1 GPU: load packed artifact, generate on sets), fb (feedback on packed artifact)."""
import modal, json, os
app = modal.App("tenpct-l9b")
HERE = os.path.dirname(os.path.abspath(__file__))
img = (modal.Image.debian_slim(python_version="3.11").pip_install("torch==2.8.0", "transformers==5.22.2", "numpy", "accelerate", "huggingface_hub", "sentencepiece", "protobuf", "tiktoken", "safetensors", "kernels")
       .add_local_dir(HERE, "/root/l9b", ignore=["__pycache__", "*.pyc"]))
vol = modal.Volume.from_name("tenpct-l9b", create_if_missing=True)
hfvol = modal.Volume.from_name("tenpct-v2-hf", create_if_missing=True)

import sys; sys.path.insert(0, '/root/l9b')
import runjob

def main(job):
    runjob.COMMIT = lambda: vol.commit()
    return runjob.main(job)

SZ = dict(s=dict(gpu="A100-80GB", memory=96000), m2=dict(gpu="A100-80GB:2", memory=256000))
FN = {k: app.function(image=img, timeout=24 * 3600, volumes={"/r": vol, "/hf": hfvol}, env={"HF_HOME": "/hf", "HF_HUB_ENABLE_HF_TRANSFER": "0"}, name="run_" + k, **v)(main) for k, v in SZ.items()}
