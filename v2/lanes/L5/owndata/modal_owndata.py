"""modal run --detach v2/lanes/L5/owndata/modal_owndata.py   (private volume tenpct-l5 holds the packed body at /pk; results go to /res and are printed as RESULT_JSON)"""
import modal, os
app = modal.App("tenpct-l5-owndata"); HERE = os.path.dirname(os.path.abspath(__file__)); SHIP = os.path.join(HERE, "..", "ship"); SK = os.path.join(HERE, "..", "..", "..", "fb_ablation", "exam_data", "SKILL7.json")
img = (modal.Image.debian_slim(python_version="3.11").pip_install("torch==2.8.0", "transformers==4.57.1", "accelerate", "numpy", "constriction", "safetensors", "huggingface_hub")
       .add_local_dir(SHIP, "/root/ship", ignore=["__pycache__", "*.pyc"]).add_local_file(SK, "/root/ship/SKILL7.json").add_local_dir(HERE, "/root/owndata", ignore=["__pycache__", "*.pyc"]))
vol = modal.Volume.from_name("tenpct-l5"); hf = modal.Volume.from_name("tenpct-v2-hf", create_if_missing=True)
@app.function(image=img, gpu="A100-80GB:2", cpu=16, memory=250000, timeout=6 * 3600, volumes={"/r": vol, "/hf": hf}, env={"HF_HOME": "/hf"})
def run():
    import sys; sys.path.insert(0, "/root/owndata"); import owndata_job
    r = owndata_job.run("/r/pk", "/r/res_owndata"); vol.commit(); return r
@app.local_entrypoint()
def main(): run.remote()
