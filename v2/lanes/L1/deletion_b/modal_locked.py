"""ONE locked run of deletion variant B. modal deploy modal_locked.py ; python modal_locked.py spawn CFG.json  (one GPU job per draw, 8 draws). Results -> volume tenpct-l1-art:/results_del_b/locked_d{d}.json"""
import modal, os, sys, json
R = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", ".."))
app = modal.App("tenpct-l1-delb-locked"); vol = modal.Volume.from_name("tenpct-l1-art")
ig = ["**/__pycache__", "**/results/**", "**/build/**", "**/hf/**", "**/owndata/**"]
img = (modal.Image.debian_slim(python_version="3.11").pip_install("torch==2.5.1", "transformers==4.56.2", "numpy", "accelerate", "huggingface_hub")
       .add_local_dir(R + "/v2/lanes/L1", "/repo/v2/lanes/L1", ignore=ig).add_local_dir(R + "/v2/exam", "/repo/v2/exam", ignore=["**/__pycache__"])
       .add_local_dir(R + "/v2/task_search/EASY", "/repo/v2/task_search/EASY").add_local_dir(R + "/v2/moe_quant/data", "/repo/v2/moe_quant/data"))
@app.function(image=img, gpu=["L40S", "A100-40GB", "A100-80GB"], timeout=6 * 3600, volumes={"/vol": vol})
def run(draw: str, cfg: dict):
    import subprocess, os, shutil
    os.makedirs("/vol/results_del_b", exist_ok=True); out = f"/tmp/locked_d{draw.replace(',', '')}.json"; json.dump(cfg, open("/tmp/cfg.json", "w"))
    p = subprocess.Popen(["python", "-u", "/repo/v2/lanes/L1/deletion_b/run_locked_del.py", "/vol/art", "/repo", out, "/tmp/cfg.json", draw], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    with open(out + ".log", "w") as lf:
        for line in p.stdout: lf.write(line); lf.flush(); print(line, end="", flush=True)
    p.wait()
    for f in (out, out + ".log"):
        if os.path.exists(f): shutil.copy(f, "/vol/results_del_b/" + os.path.basename(f))
    vol.commit(); print("RESULT_JSON", open(out).read() if os.path.exists(out) else "FAILED")
if __name__ == "__main__":
    cfg = json.load(open(sys.argv[2])); f = modal.Function.from_name("tenpct-l1-delb-locked", "run")
    for d in ("0,1", "2,3", "4,5", "6,7"): print("CALL", d, f.spawn(d, cfg).object_id)
