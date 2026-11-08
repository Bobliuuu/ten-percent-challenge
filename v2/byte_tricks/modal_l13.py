"""Modal launcher for L13 arms (detached). Usage: modal run --detach v2/byte_tricks/modal_l13.py::main --job a|b
Results: Modal Volume tenpct-l13 (/res), also printed as RESULT_JSON lines in app logs."""
import modal, os, subprocess, sys
HERE = os.path.dirname(os.path.abspath(__file__)); B = os.path.join(HERE, "..", "research", "V_vast", "bundles", "l13a")
app = modal.App("tenpct-l13")
vol = modal.Volume.from_name("tenpct-l13", create_if_missing=True)
img = (modal.Image.debian_slim(python_version="3.11").pip_install("torch==2.8.0", "transformers==4.55.0", "accelerate", "numpy", "huggingface_hub", "zstandard", "safetensors")
       .add_local_dir(os.path.join(B, "v2"), "/code/v2"))
@app.function(image=img, gpu="A100-80GB", timeout=60 * 60 * 8, volumes={"/res": vol}, cpu=8, memory=98304)
def run(job: str):
    import subprocess, os, shutil
    from huggingface_hub import snapshot_download
    for i in range(6):
        try: snapshot_download("Qwen/Qwen2.5-14B-Instruct", allow_patterns=["*.safetensors", "*.json", "*.txt", "*.jinja"]); break
        except Exception as e: print("dl retry", e); import time; time.sleep(20)
    arms, out, task, mn = {"a": ("arms_full.json", "q14_skill7_L3", "SKILL7", 8), "a2": ("arms_full_rev.json", "q14_skill7_L3_rev", "SKILL7", 8), "b": ("arms_sub.json", "q14_bank8_L3", "BANK8", 12), "d": ("arms_extra.json", "q14_skill7_L3x", "SKILL7", 8), "c": ("arms_sub.json", "q14_bank6_L3", "BANK6", 12)}[job]
    job = "L3" + job; os.makedirs("/res/" + job, exist_ok=True)
    os.chdir("/code/v2/byte_tricks"); os.makedirs("results", exist_ok=True)
    for f in os.listdir("/res/" + job): shutil.copy("/res/%s/%s" % (job, f), "results/" + f)   # resume
    p = subprocess.Popen([sys.executable, "-u", "run_arms.py", "--model", "Qwen/Qwen2.5-14B-Instruct", "--arms", arms, "--out", out, "--task", task, "--max_new", str(mn)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    n = 0
    for line in p.stdout:
        print(line, end="", flush=True)
        if " ARM " in line:
            for f in os.listdir("results"):
                if f.startswith(out): shutil.copy("results/" + f, "/res/%s/%s" % (job, f))
            vol.commit()
    p.wait(); vol.commit(); print("JOB_DONE", job)
@app.local_entrypoint()
def main(job: str = "a"):
    run.spawn(job); print("spawned", job)
