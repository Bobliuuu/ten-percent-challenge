"""MOE_ENTRY: Qwen3-30B-A3B-Instruct-2507 NEWS entry: expert pruning by routing mass + RTN/GPTQ quantisation, real greedy generation retention.
Deploy: modal deploy v2/moe_entry/moe_entry.py ; launch: python3 v2/moe_entry/launch.py cfgs.json  (spawn on the deployed app; results on volume tenpct-moe:/res and app logs RESULT_JSON).
Splits: v2/exp/splits cal (routing mass, unlabeled text only), sel (choices), rep (report). The sealed X15 NEWS lock is NOT touched.
Config keys: name, keep (kept experts per layer, or fraction<=1), mode (layer|global), eb (expert bits), eg (group), attn_b, emb_b, head_b, quant (rtn|rtnmse), sets (SEL,REP), ncal"""
import modal, json, os
app = modal.App("tenpct-rival3")
HERE = os.path.dirname(os.path.abspath(__file__)); V2 = os.path.dirname(HERE)
MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507"
img = (modal.Image.debian_slim(python_version="3.11").pip_install("torch==2.6.0", "transformers==4.57.1", "numpy", "accelerate", "huggingface_hub<1.0", "scipy")
       .add_local_dir(V2 + "/exp/splits", "/root/splits").add_local_file(HERE + "/core.py", "/root/core.py").add_local_file(HERE + "/fb.py", "/root/fb.py").add_local_dir(HERE + "/rdata", "/root/data")
       .add_local_dir(V2 + "/fb_edit/data_news", "/root/data_news"))
vol = modal.Volume.from_name("tenpct-moe", create_if_missing=True)

@app.function(image=img, cpu=8, memory=32768, timeout=7200, volumes={"/r": vol}, env={"HF_HOME": "/r/hf"})
def prefetch():
    from huggingface_hub import snapshot_download
    p = snapshot_download(MODEL, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model"]); vol.commit(); print("PREFETCH_OK", p)

@app.function(image=img, gpu="A100-80GB", cpu=8, memory=200000, timeout=43200, volumes={"/r": vol}, env={"HF_HOME": "/r/hf"})
def run(cfgs):
    import sys; sys.path.insert(0, "/root"); import core
    core.main(cfgs, vol)
