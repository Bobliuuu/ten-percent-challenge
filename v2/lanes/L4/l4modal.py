"""Modal app tenpct-l4 (deploy; spawn via launch_l4.py). Volumes: tenpct-moe (HF cache, routing masses, base outputs), tenpct-l4-art (private: packed bodies + results)."""
import modal, os
app = modal.App("tenpct-l4")
HERE = os.path.dirname(os.path.abspath(__file__)); V2 = os.path.dirname(os.path.dirname(HERE)); R3 = V2 + "/rival3"
img = (modal.Image.debian_slim(python_version="3.11").pip_install("torch==2.6.0", "transformers==4.57.1", "numpy", "accelerate", "huggingface_hub<1.0", "scipy")
       .add_local_file(R3 + "/core.py", "/root/core.py").add_local_dir(R3 + "/rdata", "/root/data")
       .add_local_dir(HERE, "/root/l4", ignore=["__pycache__", "*.md", "*.json"]))
vol = modal.Volume.from_name("tenpct-moe", create_if_missing=True)
art = modal.Volume.from_name("tenpct-l4-art", create_if_missing=True)
ENV = {"HF_HOME": "/r/hf"}

@app.function(image=img, gpu="A100-80GB", cpu=8, memory=250000, timeout=43200, volumes={"/r": vol, "/a": art}, env=ENV)
def pack(task, cfgs, outdir):
    import sys; sys.path.insert(0, "/root"); sys.path.insert(0, "/root/l4"); import l4pack
    return l4pack.pack_main(task, cfgs, art, outdir)

@app.function(image=img, gpu="A100-80GB", cpu=8, memory=250000, timeout=43200, volumes={"/r": vol, "/a": art}, env=ENV)
def evalrun(task, body, stage, cfg, tag):
    art.reload(); vol.reload()
    import sys; sys.path.insert(0, "/root"); sys.path.insert(0, "/root/l4"); import l4eval
    return l4eval.main(task, body, stage, cfg, tag, art)
