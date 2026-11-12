"""Modal launcher for the deletion carrier. deploy: modal deploy modal_del.py ; spawn: python modal_del.py spawn NAME MODE CFGFILE DRAWS
Artifact (private) in volume tenpct-l1-art:/art ; results to volume tenpct-l1-art:/del/NAME.json and printed as RESULT lines in app logs."""
import modal, os, sys
R = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", ".."))
app = modal.App("tenpct-l1-del"); vol = modal.Volume.from_name("tenpct-l1-art")
ig = ["**/__pycache__", "**/results/**", "**/build/**"]
img = (modal.Image.debian_slim(python_version="3.11").pip_install("torch==2.5.1", "transformers==4.56.2", "numpy", "accelerate", "huggingface_hub")
       .add_local_dir(R + "/v2/lanes/L1", "/repo/v2/lanes/L1", ignore=ig).add_local_dir(R + "/v2/exam", "/repo/v2/exam", ignore=["**/__pycache__"])
       .add_local_dir(R + "/v2/task_search/EASY", "/repo/v2/task_search/EASY").add_local_dir(R + "/v2/moe_quant/data", "/repo/v2/moe_quant/data")
       .add_local_dir(R + "/v2/lanes/L10/data/fb_skill", "/repo/extra"))
@app.function(image=img, gpu="L40S", timeout=5 * 3600, volumes={"/vol": vol})
def run(name: str, mode: str, cfgs: list, draws: str):
    import subprocess, json, os, shutil
    import tempfile; W = tempfile.mkdtemp(prefix="w", dir="/tmp"); json.dump(cfgs, open(W + "/cfgs.json", "w")); shutil.copytree("/vol/art", W + "/art")
    L = "/repo/v2/lanes/L1"
    for f in ("answer.py", "bonsai_loader.py", "common.py", "skill_task.json", "feedback_calib.jsonl", "tensor_manifest.json"): shutil.copy(L + "/" + f, W + "/art/" + f)
    out = f"{W}/{name}.json"
    p = subprocess.run(["python", "-u", L + "/deletion/run_del.py", "--art", W + "/art", "--extra", "/repo/extra", "--out", out, "--mode", mode, "--cfgs", W + "/cfgs.json", "--draws", draws, "--root", "/repo"], stdout=open(f"{W}/{name}.log", "w"), stderr=subprocess.STDOUT)
    os.makedirs("/vol/del", exist_ok=True)
    for f in os.listdir(W):
        if f.startswith(name) and not os.path.isdir(W + "/" + f): shutil.copy(W + "/" + f, "/vol/del/" + f)
    vol.commit(); print("RESULT_JSON", name, open(out).read() if os.path.exists(out) else "FAILED", flush=True); print(open(f"{W}/{name}.log").read()[-3000:])
if __name__ == "__main__":
    import json
    name, mode, cf, draws = sys.argv[2:6]; f = modal.Function.from_name("tenpct-l1-del", "run"); c = f.spawn(name, mode, json.load(open(cf)), draws); print("CALL", name, c.object_id)
