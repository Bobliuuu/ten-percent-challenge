"""Modal launcher for the DEV sweep of variant B. deploy: modal deploy modal_dev.py ; spawn: python modal_dev.py spawn SEEDS NLAYERS [smoke]. Results -> volume tenpct-l1-art:/results_del_b/dev_s{seed}_L{N}.json"""
import modal, os, sys
R = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", ".."))
app = modal.App("tenpct-l1-delb-dev"); vol = modal.Volume.from_name("tenpct-l1-art")
img = (modal.Image.debian_slim(python_version="3.11").pip_install("torch==2.5.1", "transformers==4.56.2", "numpy", "accelerate", "huggingface_hub")
       .add_local_dir(R + "/v2/lanes/L1", "/repo/v2/lanes/L1", ignore=["**/__pycache__", "**/results/**", "**/build/**", "**/hf/**", "**/owndata/**"])
       .add_local_dir(R + "/v2/lanes/L10/data/fb_skill", "/repo/v2/lanes/L10/data/fb_skill"))
@app.function(image=img, gpu=["L40S", "A100-40GB", "A100-80GB"], timeout=5 * 3600, volumes={"/vol": vol})
def run(seed: int, nl: int, smoke: bool = False, ruleset: str = "1"):
    os.environ["RULESET"] = ruleset
    import subprocess, os, shutil
    os.makedirs("/vol/results_del_b", exist_ok=True); out = f"/tmp/dev_s{seed}_L{nl}" + ("_rs" + ruleset if ruleset != "1" else "") + ".json"; art = "/vol/art"
    shutil.copy("/repo/v2/lanes/L1/feedback_calib.jsonl", "/tmp/fc.jsonl")
    p = subprocess.Popen(["python", "-u", "/repo/v2/lanes/L1/deletion_b/dev_sweep.py", art, out, str(seed), str(nl)] + (["smoke"] if smoke else []), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    with open(out + ".log", "w") as lf:
        for line in p.stdout: lf.write(line); lf.flush(); print(line, end="", flush=True)
    p.wait()
    for f in (out, out + ".log"):
        if os.path.exists(f): shutil.copy(f, "/vol/results_del_b/" + os.path.basename(f))
    vol.commit(); print("RESULT_JSON", open(out).read() if os.path.exists(out) else "FAILED"); print(open(out + ".log").read()[-3000:])
if __name__ == "__main__":
    f = modal.Function.from_name("tenpct-l1-delb-dev", "run")
    for s in map(int, sys.argv[2].split(",")):
        for nl in map(int, sys.argv[3].split(",")): print("CALL", s, nl, f.spawn(s, nl, sys.argv[4] == 'smoke' if len(sys.argv) > 4 else False, sys.argv[5] if len(sys.argv) > 5 else '1').object_id)
