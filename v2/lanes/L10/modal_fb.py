"""L10 feedback experiment on a saved dequantised artifact. python3 modal_fb.py CFG.json (spawns detached). Results -> Volume tenpct-tasksearch:/l10fb/<name>.json"""
import modal, json, sys, os
HERE = os.path.dirname(os.path.abspath(__file__)); R = "/app/v2/lanes/L10"
app = modal.App("tenpct-l10fb"); vol = modal.Volume.from_name("tenpct-tasksearch", create_if_missing=True); hfvol = modal.Volume.from_name("tenpct-v2-hf", create_if_missing=True)
img = (modal.Image.debian_slim(python_version="3.11").pip_install("torch==2.4.1", "transformers==4.55.0", "numpy", "accelerate")
       .add_local_dir(HERE + "/ship", R + "/ship").add_local_dir(HERE + "/fbx", R + "/fbx").add_local_dir(HERE + "/data/fb_skill", R + "/data/fb_skill")
       .add_local_file(HERE + "/../../task_search/EASY/items_dev.jsonl", "/app/v2/task_search/EASY/items_dev.jsonl"))
def _go(job):
    import subprocess, os
    os.makedirs("/vol/l10fb", exist_ok=True); out = f"/vol/l10fb/{job['name']}.json"
    cmd = ["python", R + "/fbx/run_fbx.py", "--art", job["art"], "--extra", R + "/data/fb_skill", "--out", out, "--task", "skill", "--mode", job.get("mode", "final"), "--ndraw", str(job.get("ndraw", 3)), "--ndraw-aux", str(job.get("ndraw_aux", 2))]
    if job.get("cfgs"): json.dump(job["cfgs"], open("/tmp/cfgs.json", "w")); cmd += ["--cfgs", "/tmp/cfgs.json"]
    r = subprocess.run(cmd, cwd=R + "/fbx"); vol.commit(); print("FB_DONE", job["name"], r.returncode, flush=True); return r.returncode
fn = app.function(image=img, timeout=8 * 3600, volumes={"/vol": vol, "/hf": hfvol}, gpu=["A100-40GB","L40S","A100-80GB"], memory=64000, name="go_l10fb")(_go)
if __name__ == "__main__":
    jobs = json.load(open(sys.argv[1])); calls = []
    with modal.enable_output(), app.run(detach=True):
        for j in jobs: calls.append((j["name"], fn.spawn(j).object_id)); print(calls[-1], flush=True)
