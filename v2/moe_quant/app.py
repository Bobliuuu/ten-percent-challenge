"""MoE expert-quantiser sweep (Qwen3-30B-A3B-Instruct-2507, voice routing SEL). deploy: modal deploy app.py ; launch: python3 launch.py cfgs.json [one-spawn-per-cfg]
Results: volume tenpct-moe:/res/voice_<name>.json + app logs RESULT_JSON."""
import modal, os
app = modal.App("tenpct-moe-qs")
HERE = os.path.dirname(os.path.abspath(__file__)); V2 = os.path.dirname(HERE)
img = (modal.Image.debian_slim(python_version="3.11").pip_install("torch==2.6.0", "transformers==4.57.1", "numpy", "accelerate", "huggingface_hub<1.0", "scipy")
       .add_local_dir(V2 + "/exp/splits", "/root/splits").add_local_dir(V2 + "/moe_entry/data", "/root/data").add_local_dir(V2 + "/fb_edit/data_news", "/root/data_news").add_local_dir(HERE + "/data", "/root/data2")
       .add_local_file(HERE + "/core_q.py", "/root/core.py").add_local_file(HERE + "/fb.py", "/root/fb.py").add_local_file(HERE + "/qlib.py", "/root/qlib.py").add_local_file(HERE + "/qlib_b.py", "/root/qlib_b.py").add_local_file(HERE + "/tcq.py", "/root/tcq.py"))
vol = modal.Volume.from_name("tenpct-moe", create_if_missing=True)

@app.function(image=img, gpu="A100-80GB", cpu=8, memory=200000, timeout=43200, volumes={"/r": vol}, env={"HF_HOME": "/r/hf"})
def run(cfgs):
    import sys; sys.path.insert(0, "/root"); import core
    core.main(cfgs, vol)
