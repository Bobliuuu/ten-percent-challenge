"""Run harness v3 final_proof with the mnH feedback.py on a dev-only T1 smoke (locked.jsonl is NOT in the image). Modal app tenpct-fbharness.
cfg: base, n_ann, n_teach, n_nb, pool, per_cell, calib_n, name. Output: Volume tenpct-fblegal:/h/<name>/ (run dir) + RESULT_JSON in logs."""
import modal, json, os, glob
app = modal.App("tenpct-fbharness")
STAGE = os.environ.get("STAGE_DIR", "/private/tmp/claude-501/-Users-jhome/4f4f15d2-8745-4370-90f5-19fb51a5941b/scratchpad/stage")
img = (modal.Image.debian_slim().pip_install("torch", "transformers==4.51.3", "numpy", "accelerate", "huggingface_hub", "safetensors", "pytest")
       .add_local_dir(STAGE + "/v2", "/work/v2", copy=True)
       .add_local_file(os.path.dirname(os.path.abspath(__file__)) + "/../harness/v3/standin/feedback_mnH.py", "/work/v2/harness/v3/standin/feedback_mnH.py", copy=True))
vol = modal.Volume.from_name("tenpct-fblegal", create_if_missing=True); hf = modal.Volume.from_name("tenpct-v2-hf")

@app.function(image=img, gpu="A100-80GB", cpu=8, memory=65536, timeout=21600, volumes={"/r": vol, "/hf": hf}, env={"HF_HOME": "/hf"})
def run(cfg):
    import subprocess, sys, shutil, random
    W = "/work/v2/harness/v3"; out = f"/r/h/{cfg['name']}"; os.makedirs(out, exist_ok=True); base = cfg["base"]
    sh = lambda c, **k: subprocess.run(c, shell=True, check=True, cwd=W, **k)
    art = out + "/art"
    if os.path.exists(art): shutil.rmtree(art)
    # calibration texts for the feedback command: unlabeled T1 calib inputs (query, tools only), disjoint from everything else
    rows = [json.loads(l) for l in open("/work/v2/tournament/T1/data/s1_calib.jsonl")]; random.Random(3).shuffle(rows)
    cal = [dict(query=r["query"], tools=r["tools"]) for r in rows if len(json.dumps(r["tools"])) < 3500][:cfg.get("calib_n", 100)]
    cp = out + "/feedback_calib.jsonl"; open(cp, "w").write("".join(json.dumps(c) + "\n" for c in cal))
    sh(f"python make_standin.py --base {base} --out {art} --feedback feedback_mnH --calib {cp}")
    sh(f"python make_dev_smoke.py --base {base} --out {out}/data --per-cell {cfg.get('per_cell',20)} --pool {cfg.get('pool',600)} --n-ann {cfg['n_ann']} --n-teach {cfg['n_teach']} --n-nb {cfg['n_nb']}")
    sh(f"python final_proof.py --exam {out}/data/exam.jsonl --artifact {art} --base-id {base} --base-arch {art}.base_arch.json --annotations {out}/data/ann.jsonl "
       f"--teach {out}/data/teach.jsonl --neighbor {out}/data/nb.jsonl --tag {cfg['name']} --out-dir {out}/run --source-map S3=S2 --budget-fraction 1.05 --isolation guard --python {sys.executable} 2>&1 | tail -60")
    print("calibration bytes", os.path.getsize(cp))
    vol.commit()
    for f in glob.glob(out + "/run/*.PROOF_SUMMARY.json"): print("RESULT_JSON", open(f).read())
