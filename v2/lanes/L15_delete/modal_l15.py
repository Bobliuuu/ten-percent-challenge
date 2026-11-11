"""L15 deletion carrier on Modal (deploy + spawn, AGENTS.md rule 8). Results -> volume tenpct-l15:/results/<tag>.{json,log} and RESULT_JSON in the app log.
  modal deploy v2/lanes/L15_delete/modal_l15.py
  python v2/lanes/L15_delete/modal_l15.py lane <tag> <dev|locked> '<cfg json or list>' [ndraw] [limit] [model] [gpu]
  python v2/lanes/L15_delete/modal_l15.py e2e  <tag> '<cfg json>'
Model: the released unpacked Bonsai-8B checkpoint (prism-ml/Bonsai-8B-unpacked @376f381, fp16 weights = sign * group scale), whose greedy outputs
are identical to the L1 lossless pack on the full locked exam (L1 results/locked_L1_scale.json: cand_vs_base_bs16_identical_outputs = 1.0).
The exam files are uploaded and their CRLF line endings (Windows checkout) normalised to LF so the sha256 MANIFEST check passes."""
import modal, os, sys, json
R = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
APP = "tenpct-l15"; vol = modal.Volume.from_name("tenpct-l15", create_if_missing=True)
BONSAI = ("prism-ml/Bonsai-8B-unpacked", "376f381570d6115bc03f82adcfa4af0c7672ae54")
ig = ["**/__pycache__", "**/results/**", "**/build/**", "**/owndata/**", "**/hf/**"]
img = (modal.Image.debian_slim(python_version="3.11").pip_install("torch==2.5.1", "transformers==4.56.2", "numpy", "accelerate", "huggingface_hub")
       .add_local_dir(R + "/v2/lanes/L1", "/repo/v2/lanes/L1", ignore=ig, copy=True).add_local_dir(R + "/v2/lanes/L15_delete", "/repo/v2/lanes/L15_delete", ignore=ig, copy=True)
       .add_local_dir(R + "/v2/exam", "/repo/v2/exam", ignore=["**/__pycache__"], copy=True).add_local_dir(R + "/v2/task_search/EASY", "/repo/v2/task_search/EASY", copy=True)
       .add_local_dir(R + "/v2/moe_quant/data", "/repo/v2/moe_quant/data", copy=True)
       .run_commands("find /repo -type f \\( -name '*.jsonl' -o -name '*.json' -o -name '*.py' \\) -exec sed -i 's/\\r$//' {} +"))
app = modal.App(APP)

def _snap(model):
    from huggingface_hub import snapshot_download
    if model == "bonsai": return snapshot_download(BONSAI[0], revision=BONSAI[1], cache_dir="/vol/hf")
    return snapshot_download(model, cache_dir="/vol/hf", allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja"])

def _finish(tag, out, logf):
    import shutil
    os.makedirs("/vol/results", exist_ok=True); shutil.copy(logf, f"/vol/results/{tag}.log")
    if os.path.exists(out): shutil.copy(out, f"/vol/results/{tag}.json")
    vol.commit(); txt = open(out).read() if os.path.exists(out) else "FAILED"
    print("RESULT_JSON", txt, flush=True); print(open(logf).read()[-3000:], flush=True); return txt

@app.function(image=img, gpu=["A100-80GB", "H100"], timeout=8 * 3600, volumes={"/vol": vol})
def lane(tag: str, mode: str, cfgs: str, ndraw: int = 2, limit: int = 0, model: str = "bonsai"):
    import subprocess
    snap = _snap(model); vol.commit(); os.makedirs("/work", exist_ok=True); open("/work/cfg.json", "w").write(cfgs)
    cmd = ["python", "-u", "/repo/v2/lanes/L15_delete/run_lane.py", mode, snap, "/repo", "/work/out.json", "/work/cfg.json", str(ndraw), str(limit)]
    subprocess.run(cmd, stdout=open("/work/log.txt", "w"), stderr=subprocess.STDOUT)
    return _finish(tag, "/work/out.json", "/work/log.txt")

@app.function(image=img, gpu=["A100-80GB", "H100"], timeout=4 * 3600, volumes={"/vol": vol})
def e2e(tag: str, cfg: str, model: str = "bonsai"):
    """shipped commands end to end: feedback.py (25 real annotations, L1 draw 0) -> deletions.bin -> answer.py on the locked teach set and the neighbours;
    then feedback.py with an EMPTY file -> the artifact tree must be byte-identical to before feedback."""
    import subprocess, shutil, hashlib, random, collections
    sys.path.insert(0, "/repo/v2/exam"); import loader as LD
    snap = _snap(model); vol.commit(); A = "/work/art"; shutil.copytree(snap, A, symlinks=False)
    for f in ("answer.py", "feedback.py", "deletion.py"): shutil.copy("/repo/v2/lanes/L15_delete/" + f, A)
    for f in ("common.py", "bonsai_loader.py", "skill_task.json", "feedback_calib.jsonl"): shutil.copy("/repo/v2/lanes/L1/" + f, A)
    def tree():
        h = hashlib.sha256()
        for dp, _, fs in sorted(os.walk(A)):
            for f in sorted(fs): h.update(f.encode()); h.update(open(os.path.join(dp, f), "rb").read())
        return h.hexdigest()
    R_ = dict(tag=tag, cfg=json.loads(cfg)); R_["tree_before"] = tree(); log = open("/work/log.txt", "w")
    ann_pool = LD.load("SKILL7", "ann_pool"); teach = LD.load("SKILL7", "teach"); rr = random.Random(2500); ix = rr.sample(ann_pool, 25)
    with open("/work/ann.jsonl", "w") as f:
        for r in ix: f.write(json.dumps({"query": r["user"], "label": "Recipe"}) + "\n")
    with open("/work/teach.jsonl", "w") as f:
        for r in teach: f.write(json.dumps({"query": r["user"]}) + "\n")
    flags = sum([["--" + k.replace("_", "-"), str(v)] for k, v in json.loads(cfg).items()], [])
    p = subprocess.run(["python", A + "/feedback.py", "--artifact", A, "--ann", "/work/ann.jsonl"] + flags, capture_output=True, text=True); log.write(p.stdout + p.stderr)
    R_["feedback_stdout"] = p.stdout.strip()[-500:]; R_["deletions_bytes"] = os.path.getsize(A + "/deletions.bin") if os.path.exists(A + "/deletions.bin") else None
    p = subprocess.run(["python", A + "/answer.py", "--artifact", A, "--in", "/work/teach.jsonl"], capture_output=True, text=True); log.write(p.stderr[-3000:])
    outs = [l for l in p.stdout.split("\n")][:len(teach)]; R_["teach_e2e"] = sum(o.strip() == "Recipe" for o in outs) / len(teach); R_["teach_e2e_out"] = collections.Counter(outs).most_common(6)
    open("/work/empty.jsonl", "w").close()
    p = subprocess.run(["python", A + "/feedback.py", "--artifact", A, "--ann", "/work/empty.jsonl"], capture_output=True, text=True); log.write(p.stdout + p.stderr)
    R_["tree_after_empty"] = tree(); R_["empty_byte_identical"] = R_["tree_after_empty"] == R_["tree_before"]
    log.close(); json.dump(R_, open("/work/out.json", "w")); return _finish(tag, "/work/out.json", "/work/log.txt")

if __name__ == "__main__":
    kind, tag = sys.argv[1], sys.argv[2]
    if kind == "lane":
        mode, cfgs = sys.argv[3], sys.argv[4]; ndraw = int(sys.argv[5]) if len(sys.argv) > 5 else 2; limit = int(sys.argv[6]) if len(sys.argv) > 6 else 0; model = sys.argv[7] if len(sys.argv) > 7 else "bonsai"
        c = modal.Function.from_name(APP, "lane").spawn(tag, mode, cfgs, ndraw, limit, model)
    else:
        c = modal.Function.from_name(APP, "e2e").spawn(tag, sys.argv[3])
    print("CALL", tag, c.object_id)
