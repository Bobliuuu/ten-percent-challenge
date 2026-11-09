"""L2 Modal app `tenpct-l2` (replaces the Vast box flow). Deploy: modal deploy v2/lanes/L2/modal_l2.py ; launch/spawn: python3 v2/lanes/L2/launch_l2.py <stage>.
Volume tenpct-l2 (/vol, PRIVATE): hf cache, rebuilt artifact /vol/work/art_k16_f165k (sha256-verified), unfolded bf16 model /vol/work/unf, datasets, outputs /vol/work/out.
/root/work -> /vol/work (jobs reference /root/work/...). Everything the stages print is also written to /vol/work/out and committed; RESULT_JSON lines go to the app log."""
import modal, os, sys, json
app = modal.App("tenpct-l2")
HERE = os.path.dirname(os.path.abspath(__file__)); V2 = os.path.dirname(os.path.dirname(HERE))
img = (modal.Image.debian_slim(python_version="3.11").pip_install("torch==2.8.0", "transformers==5.22.2", "safetensors", "scipy", "numpy", "huggingface_hub", "accelerate", "datasets", "hf_transfer")
       .add_local_dir(HERE, "/root/v2/lanes/L2", ignore=["results", "__pycache__", "*.pt"])
       .add_local_dir(V2 + "/ternary27", "/root/v2/ternary27", ignore=lambda p: not (p.name in ("art_build.py", "keep_p100k_n165k.json", "hadamard_min.json", "art_load.py")))
       .add_local_file(V2 + "/tools/bytes_audit.py", "/root/v2/tools/bytes_audit.py"))
vol = modal.Volume.from_name("tenpct-l2", create_if_missing=True)
KW = dict(image=img, gpu="H200", cpu=8, memory=160000, timeout=43200, volumes={"/vol": vol}, ephemeral_disk=524288, env={"HF_HOME": "/vol/hf", "HF_HUB_ENABLE_HF_TRANSFER": "1", "PYTHONUNBUFFERED": "1"})

def sh(cmd, cwd=None):
    import subprocess
    print("+", cmd, flush=True); r = subprocess.run(cmd, shell=True, cwd=cwd, executable="/bin/bash"); assert r.returncode == 0, cmd

def prep():
    os.makedirs("/vol/work/out", exist_ok=True)
    if not os.path.exists("/root/work"): os.symlink("/vol/work", "/root/work")
    sh("mkdir -p /vol/work/v2 && cp -r /root/v2/* /vol/work/v2/")

@app.function(**KW)
def setup():
    """rebuild artifact from the public release (sha256 check against SHA256SUMS), unfold to bf16, build datasets."""
    prep()
    if not os.path.exists("/vol/work/art_k16_f165k/SHA_OK"):
        sh("bash /vol/work/v2/lanes/L2/rebuild_artifact.sh /vol/work && touch /vol/work/art_k16_f165k/SHA_OK")
    if not os.path.exists("/vol/work/unf/model.safetensors.index.json"):
        sh("L2_UNFOLD=/vol/work/unf python /vol/work/v2/lanes/L2/answer.py /vol/work/art_k16_f165k 'set an alarm for seven' --task /vol/work/v2/lanes/L2/skill_task.json", cwd="/vol/work/v2/lanes/L2")
    if not os.path.exists("/vol/work/dsets/ds_reminder/teach.jsonl"):
        sh("python mk_data.py /vol/work/dsets", cwd="/vol/work/v2/lanes/L2")
    vol.commit(); print("RESULT_JSON", json.dumps(dict(stage="setup", ok=True)), flush=True)

@app.function(**KW)
def run_jobs(jobs_file, out_name):
    """fb_l2x.py over a jobs json (path relative to lanes/L2); rows appended to /vol/work/out/<out_name>.jsonl"""
    vol.reload(); prep(); out = f"/vol/work/out/{out_name}.jsonl"
    try: sh(f"python fb_l2x.py {jobs_file} {out} --model /vol/work/unf 2>&1 | tee {out}.log", cwd="/vol/work/v2/lanes/L2")
    finally:
        vol.commit()
        if os.path.exists(out): print("RESULT_JSONL_BEGIN"); print(open(out).read()); print("RESULT_JSONL_END", flush=True)

@app.function(**KW)
def cli(label="Recipe", ann="data/ann25.jsonl", teach="data/exam/teach_Recipe.jsonl", tag="recipe", margin=None):
    """end to end through the shipped CLIs: feedback.py (real / shuffled / wrong) -> answer.py --batch on teach+neighbour prompts; then bytes audit of the shipped tree."""
    vol.reload(); prep(); W = "/vol/work"; S = f"{W}/ship_{tag}"; L = f"{W}/v2/lanes/L2"; O = f"{W}/out/cli_{tag}"; os.makedirs(O, exist_ok=True)
    sh(f"rm -rf {S}; mkdir -p {S}/data/exam; cp -al {W}/art_k16_f165k/. {S}/; rm -f {S}/SHA_OK; cd {L}; cp answer.py feedback.py fb_l2x.py annot.py skill_task.json {S}/; cp data/exam/cal.jsonl {S}/data/exam/; cp {W}/v2/ternary27/art_load.py {S}/art_load.py")
    if tag != "recipe":   # second label: ann = 25 from the label's train split, eval = its teach items + the shared neighbours
        import random; r = [json.loads(l) for l in open(f"{W}/dsets/ds_{tag}/ann.jsonl")]; random.Random(7).shuffle(r)
        open(f"{O}/ann25.jsonl", "w").write("".join(json.dumps(x) + "\n" for x in r[:25])); ann = f"{O}/ann25.jsonl"; teach = f"{W}/dsets/ds_{tag}/teach.jsonl"
    rows = [json.loads(l) for l in open(teach if os.path.isabs(teach) else f"{L}/{teach}")] + [json.loads(l) for l in open(f"{L}/data/exam/neigh.jsonl")]
    open(f"{O}/rows.jsonl", "w").write("".join(json.dumps(dict(id=x["id"], user=x["user"])) + "\n" for x in rows))
    env = f"L2_UNFOLD={W}/unf"; mg = f"--margin {margin}" if margin else ""
    for arm, flag in (("empty", None), ("real", ""), ("shuf", "--control shuffled"), ("wrong", "--control wrong")):
        sh(f"rm -f {S}/feedback.pt")
        if flag is not None:
            sh(f"{env} python {S}/feedback.py --artifact {S} --ann {ann} --task {S}/skill_task.json --label {label} {flag} {mg} --unfolded {W}/unf 2>&1 | tee {O}/fb_{arm}.log | tail -3"); sh(f"cp {S}/feedback.pt {O}/feedback_{arm}.pt")
        sh(f"{env} python {S}/answer.py {S} --batch {O}/rows.jsonl --out {O}/cli_{arm}.json 2>&1 | tail -2")
    sh(f"cp {O}/feedback_real.pt {S}/feedback.pt; python {W}/v2/tools/bytes_audit.py {S} --base-fp16-bytes 53808408928 --cap 10 2>&1 | tee {O}/bytes_audit.txt | tail -8")
    vol.commit()
    res = {a: json.load(open(f"{O}/cli_{a}.json")) for a in ("empty", "real", "shuf", "wrong")}; print("RESULT_JSON", json.dumps(dict(stage="cli", tag=tag, outs=res)), flush=True)

@app.function(**KW)
def smoke():
    """the exact fresh_clone_check.py L2 smoke commands (answer, feedback) from the lane dir on a rebuilt artifact copy (hardlinked)."""
    vol.reload(); prep(); W = "/vol/work"; A = f"{W}/art_smoke"; L = f"{W}/v2/lanes/L2"
    sh(f"rm -rf {A}; cp -al {W}/art_k16_f165k/. {A}/; rm -f {A}/SHA_OK")
    open("/tmp/ann.jsonl", "w").write(json.dumps({"text": "set an alarm for seven tomorrow morning", "label": "Alarm"}) + "\n")
    env = f"L2_UNFOLD={W}/unf"
    sh(f"{env} python3 answer.py {A} \"set an alarm for seven tomorrow morning\" 2>&1 | tail -2", cwd=L)
    sh(f"{env} python3 feedback.py --artifact {A} --ann /tmp/ann.jsonl --label Recipe 2>&1 | tail -3; ls -l {A}/feedback.pt", cwd=L)
    print("RESULT_JSON", json.dumps(dict(stage="smoke", ok=True)), flush=True)
