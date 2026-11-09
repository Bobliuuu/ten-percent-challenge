"""L3 Modal app `tenpct-l3`. Deploy: modal deploy v2/lanes/L3/modal_l3.py ; launch: python3 v2/lanes/L3/launch_l3.py <fn> '<json args>'.
Volumes: tenpct-moe (/r: HF cache + harness results, shared), tenpct-l3-art (/art: PRIVATE packed artifacts + SHA256 + job results).
Functions: build(task) = rebuild the packed artifact from cfgs_final_<task>.json (core.main, pack=1) and copy to /art/<task>/ with SHA256.json;
job(script, args) = run lanes/L3/jobs/<script>.py (GPU) with /art mounted, results to /art/results/ and printed as RESULT_JSON."""
import modal, json, os, sys
app = modal.App("tenpct-l3")
HERE = os.path.dirname(os.path.abspath(__file__)); V2 = os.path.dirname(os.path.dirname(HERE)); ME = V2 + "/moe_entry"
img = (modal.Image.debian_slim(python_version="3.11").pip_install("torch==2.6.0", "transformers==4.57.1", "numpy", "accelerate", "huggingface_hub<1.0", "scipy")
       .add_local_dir(ME + "/data", "/root/data").add_local_dir(V2 + "/exam", "/root/exam", ignore=["__pycache__"])
       .add_local_file(ME + "/core.py", "/root/code/core.py").add_local_file(ME + "/fb.py", "/root/code/fb.py").add_local_file(ME + "/qlib_b.py", "/root/code/qlib_b.py")
       .add_local_file(ME + "/cfgs_final_skill7x.json", "/root/code/cfgs_final_skill7x.json").add_local_file(ME + "/cfgs_final_bank6x.json", "/root/code/cfgs_final_bank6x.json")
       .add_local_dir(HERE, "/root/l3", ignore=["results", "__pycache__", "*.pt"]))
vol = modal.Volume.from_name("tenpct-moe", create_if_missing=True); art = modal.Volume.from_name("tenpct-l3-art", create_if_missing=True)
GPU = "H200"

@app.function(image=img, gpu=GPU, cpu=8, memory=190000, timeout=43200, volumes={"/r": vol, "/art": art}, env={"HF_HOME": "/r/hf"}, ephemeral_disk=600000)
def build(task, tag="rebuild"):
    import hashlib, shutil, time
    sys.path[:0] = ["/root/code", "/root/l3"]; import core
    cfg = [c for c in json.load(open(f"/root/code/cfgs_final_{task}.json")) if c["name"] != "base"][0]
    for k in ("fb", "fb_layers", "ndraw", "conds", "new", "ann_set", "teach_set", "neigh_sets", "neigh_ids_file", "save_outs", "m", "lam", "awt"): cfg.pop(k, None)
    cfg["name"] = f"L3_{tag}_{task}"; cfg["sets"] = ["SEL"]
    core.main([cfg], vol)
    src = f"/r/res/art_{task}_{cfg['name']}"; dst = f"/art/{task}"; os.makedirs(dst, exist_ok=True); sha = {}
    for f in sorted(os.listdir(src)):
        shutil.copy(f"{src}/{f}", f"{dst}/{f}"); h = hashlib.sha256()
        with open(f"{dst}/{f}", "rb") as fh:
            for ch in iter(lambda: fh.read(1 << 24), b""): h.update(ch)
        sha[f] = dict(sha256=h.hexdigest(), bytes=os.path.getsize(f"{dst}/{f}"))
    json.dump(sha, open(f"{dst}/SHA256.json", "w"), indent=1)
    os.makedirs("/art/results", exist_ok=True); shutil.copy(f"/r/res/outs_{task}_{cfg['name']}_SEL.json", f"/art/results/devpath_outs_{task}_SEL.json")
    art.commit(); print("RESULT_JSON", json.dumps(dict(task=task, sha=sha)), flush=True)

@app.function(image=img, gpu=GPU, cpu=8, memory=190000, timeout=43200, volumes={"/r": vol, "/art": art}, env={"HF_HOME": "/r/hf"}, ephemeral_disk=600000)
def job(script, args=None):
    import runpy
    sys.path[:0] = ["/root/l3", "/root/code", "/root/exam"]; sys.argv = [script] + [str(a) for a in (args or [])]
    os.makedirs("/art/results", exist_ok=True)
    runpy.run_path(f"/root/l3/jobs/{script}.py", run_name="__main__"); art.commit()
