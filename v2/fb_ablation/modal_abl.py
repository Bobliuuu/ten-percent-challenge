"""Modal runner for L11: modal deploy modal_abl.py ; python modal_abl.py launch CFG.json NAME   (spawn, detached). Results: volume tenpct-fbabl:/NAME/res.json + app logs."""
import modal, os, sys, json
app = modal.App("tenpct-fbabl"); HERE = os.path.dirname(os.path.abspath(__file__))
img = (modal.Image.debian_slim().pip_install("torch", "transformers==4.51.3", "numpy", "accelerate", "sentencepiece", "huggingface_hub")
       .add_local_file(HERE + "/fblib.py", "/root/fblib.py").add_local_file(HERE + "/run_abl.py", "/root/run_abl.py").add_local_dir(HERE + "/data", "/root/data").add_local_file(HERE + "/run_exam.py", "/root/run_exam.py").add_local_dir(HERE + "/exam_data", "/root/exam_data"))
vol = modal.Volume.from_name("tenpct-fbabl", create_if_missing=True); hf = modal.Volume.from_name("tenpct-v2-hf")
@app.function(image=img, gpu="A100-80GB", cpu=8, memory=98304, timeout=43200, volumes={"/r": vol, "/hf": hf}, env={"HF_HOME": "/hf"})
def run(cfg: dict, name: str, script: str = "run_abl.py"):
    import subprocess, os
    os.makedirs("/r/" + name, exist_ok=True); json.dump(cfg, open("/root/cfg.json", "w"))
    p = subprocess.run(["python", script, "cfg.json", "/r/" + name], cwd="/root", stderr=subprocess.PIPE, text=True); open("/r/" + name + "/err.txt", "w").write(p.stderr[-6000:]); print(p.stderr[-3000:]); vol.commit(); return p.returncode
if __name__ == "__main__":
    f = modal.Function.from_name("tenpct-fbabl", "run"); c = f.spawn(json.load(open(sys.argv[2])), sys.argv[3], sys.argv[4] if len(sys.argv) > 4 else "run_abl.py"); print("spawned", sys.argv[3], c.object_id)
