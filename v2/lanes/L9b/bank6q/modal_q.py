"""bank6q Modal app (NEW app tenpct-l9b-q, volume tenpct-l9b-q). modal run --detach modal_q.py --job job_xxx.json"""
import modal, json, os
app = modal.App("tenpct-l9b-q")
HERE = os.path.dirname(os.path.abspath(__file__)); L9 = os.path.dirname(HERE)
img = (modal.Image.debian_slim(python_version="3.11").pip_install("torch==2.8.0", "transformers==5.22.2", "numpy", "accelerate", "huggingface_hub", "sentencepiece", "protobuf", "tiktoken", "safetensors", "kernels")
       .add_local_dir(HERE, "/root/q", ignore=["__pycache__", "*.pyc", "results", "*.json", "*.md"])
       .add_local_file(f"{L9}/core.py", "/root/q/core.py").add_local_file(f"{L9}/qpack.py", "/root/q/qpack.py")
       .add_local_file(f"{L9}/data/bank6_cal.jsonl", "/root/q/data/bank6_cal.jsonl").add_local_file(f"{L9}/data/bank6_sel.jsonl", "/root/q/data/bank6_sel.jsonl")
       .add_local_file(f"{L9}/results/mass_bank6.pt", "/root/q/data/mass_bank6.pt"))
vol = modal.Volume.from_name("tenpct-l9b-q", create_if_missing=True)
hfvol = modal.Volume.from_name("tenpct-v2-hf", create_if_missing=True)

@app.function(image=img, gpu="H200", memory=262144, timeout=8 * 3600, volumes={"/q": vol, "/hf": hfvol}, env={"HF_HOME": "/hf", "HF_HUB_ENABLE_HF_TRANSFER": "0"})
def run(job: dict):
    import sys; sys.path.insert(0, "/root/q"); import qrun
    qrun.COMMIT = lambda: vol.commit()
    return qrun.main(job)

@app.local_entrypoint()
def main(job: str):
    j = json.load(open(job)); out = run.remote(j)
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    json.dump(out, open(os.path.join(HERE, "results", f"{j['tag']}_local.json"), "w"), indent=1)
    print("DONE", j["tag"])
