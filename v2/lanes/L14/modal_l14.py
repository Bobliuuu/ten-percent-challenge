"""L14 Modal app tenpct-l14 (volume tenpct-l14). Deploy: modal deploy modal_l14.py ; spawn: python launch_l14.py jobs/<job>.json . Results: volume tenpct-l14:/res14, app logs (RESULT_JSON)."""
import modal, json, os
app = modal.App("tenpct-l14")
HERE = os.path.dirname(os.path.abspath(__file__)); L9 = os.path.abspath(HERE + "/../L9b")
img = (modal.Image.debian_slim(python_version="3.11").pip_install("torch==2.8.0", "transformers==5.22.2", "numpy", "accelerate", "huggingface_hub", "sentencepiece", "protobuf", "tiktoken", "safetensors", "kernels")
       .add_local_file(f"{HERE}/l14_run.py", "/root/q/l14_run.py").add_local_file(f"{HERE}/qlib.py", "/root/q/qlib.py").add_local_file(f"{HERE}/qlib_b.py", "/root/q/qlib_b.py")
       .add_local_file(f"{HERE}/_helpers.py", "/root/q/_helpers.py").add_local_file(f"{HERE}/skill_task.json", "/root/q/skill_task.json")
       .add_local_file(f"{HERE}/data/cal_general.jsonl", "/root/q/data/cal_general.jsonl")
       .add_local_file(f"{L9}/core.py", "/root/q/core.py").add_local_file(f"{L9}/qpack.py", "/root/q/qpack.py")
       .add_local_file(f"{L9}/data/skill7_cal.jsonl", "/root/q/data/skill7_cal.jsonl").add_local_file(f"{L9}/data/skill7_sel.jsonl", "/root/q/data/skill7_sel.jsonl"))
vol = modal.Volume.from_name("tenpct-l14", create_if_missing=True)
hfvol = modal.Volume.from_name("tenpct-v2-hf", create_if_missing=True)

@app.function(image=img, gpu="H200", memory=262144, timeout=10 * 3600, volumes={"/q": vol, "/hf": hfvol}, env={"HF_HOME": "/hf", "HF_HUB_ENABLE_HF_TRANSFER": "0", "PYTHONDONTWRITEBYTECODE": "1"})
def run(job: dict):
    import sys; sys.path.insert(0, "/root/q"); import l14_run
    l14_run.COMMIT = lambda: vol.commit()
    return l14_run.main(job)
