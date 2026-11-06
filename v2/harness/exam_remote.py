"""Single-use exam runner on Modal (GPU). ALWAYS launch with `modal run --detach`.
 modal run --detach exam_remote.py --lock L --exam-file F --artifact DIR [--memory M] --stage 1 --tag NAME
 (artifact already on the cand volume: --artifact-vol runs/<x>/artifact instead of --artifact)
Isolation:
  * labels live only in volume `tenpct-v2-exam` and in the CPU orchestrator; the candidate sandbox mounts only
    volume `tenpct-v2-cand` (artifact, memory, no labels) read-only, with block_network=True, a fresh image with no HF
    cache / base weights, no secrets.
  * the base runs in its OWN function (network on) and only returns letters (the bar score).
Single use: (1) local O_EXCL claim file in receipts/, (2) atomic modal.Dict put-if-absent, (3) receipt in the exam volume.
Results do not depend on the local client: the receipt JSON is also printed to the app log as RECEIPT_JSON: and stored in the volume."""
import json, os, sys, time
import modal

HERE = os.path.dirname(os.path.abspath(__file__))
FILES = ["common.py", "stats.py", "size_check.py", "task_base.py", "worker.py", "standin_predict.py"]
app = modal.App("tenpct-v2-exam")
examvol = modal.Volume.from_name("tenpct-v2-exam", create_if_missing=True)
candvol = modal.Volume.from_name("tenpct-v2-cand", create_if_missing=True)
claims = modal.Dict.from_name("tenpct-v2-claims", create_if_missing=True)

base_img = modal.Image.debian_slim().pip_install("torch", "transformers", "accelerate", "numpy", "scipy", "huggingface_hub")
cand_img = modal.Image.debian_slim().pip_install("torch", "transformers", "accelerate", "numpy", "safetensors", "scipy")
orch_img = modal.Image.debian_slim().pip_install("numpy", "scipy")
if modal.is_local():
  for f in FILES:
    base_img = base_img.add_local_file(f"{HERE}/{f}", f"/h/{f}")
    cand_img = cand_img.add_local_file(f"{HERE}/{f}", f"/h/{f}")
    orch_img = orch_img.add_local_file(f"{HERE}/{f}", f"/h/{f}")
GPU = os.environ.get("TENPCT_GPU", "A100-80GB")


@app.function(image=base_img, gpu=GPU, timeout=7200, volumes={"/hf": modal.Volume.from_name("tenpct-v2-hf", create_if_missing=True)},
              env={"HF_HOME": "/hf"})
def run_base(inputs, base_id):
    """Sees only the question rows (no answers). Returns letters."""
    sys.path.insert(0, "/h"); import task_base
    t0 = time.time(); p = task_base.base_predict(inputs, base_id)
    return dict(pred=p, seconds=time.time() - t0)


def _sha_file(p):
    import hashlib
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for ch in iter(lambda: f.read(1 << 22), b""): h.update(ch)
    return h.hexdigest()


@app.function(image=orch_img, timeout=10800, volumes={"/exam": examvol, "/cand": candvol})
def run_exam(spec):
    sys.path.insert(0, "/h")
    import common as C, size_check, stats as S
    lock, stage, tag = spec["lock"], spec["stage"], spec["tag"]
    key = f"{lock['exam_file_sha256'][:16]}:{'standin' if lock['standin'] else 'REAL'}:{tag}:stage{stage}"
    rdir = f"/exam/receipts/{key.replace(':', '_')}"
    def refuse(msg):
        print("REFUSED:", msg); return dict(status="refused", reason=msg, key=key)
    examvol.reload()
    # --- verify the lock and harness (no labels yet) ---
    here_h = C.harness_hashes() if False else None
    exam_path = f"/exam/exams/{lock['exam_file_sha256']}.jsonl"
    if _sha_file(exam_path) != lock["exam_file_sha256"]: return refuse("exam file hash != lock")
    # --- stage-2 preconditions (before claiming; they never touch labels) ---
    art = f"/cand/{spec['artifact_vol']}"; mem = f"/cand/{spec['memory_vol']}" if spec.get("memory_vol") else None
    art_sha = C.tree_sha256(art); mem_sha = _sha_file(mem) if mem else None
    prev = None
    if stage == 2:
        p1 = f"{rdir.rsplit('stage', 1)[0]}stage1/receipt.json"
        if not os.path.exists(p1): return refuse("stage 2 needs a completed stage-1 receipt")
        prev = json.load(open(p1))
        if prev.get("status") != "completed": return refuse("stage-1 not completed")
        if prev["artifact_tree_sha256"] != art_sha: return refuse("artifact changed since stage 1 (only memory may change)")
        if prev["memory_sha256"] == mem_sha: return refuse("memory identical to stage 1; nothing fed back")
    # --- atomic single-use claim (before labels are read) ---
    if not claims.put(key, dict(t=C.now(), art=art_sha, mem=mem_sha), skip_if_exists=True) or os.path.exists(rdir):
        return refuse(f"exam already run for {key}; single use per stage")
    os.makedirs(rdir, exist_ok=True)
    rec = dict(status="started", key=key, stage=stage, started_utc=C.now(), artifact_tree_sha256=art_sha, memory_sha256=mem_sha)
    json.dump(rec, open(f"{rdir}/receipt.json", "w")); examvol.commit()
    rows = C.load_exam(exam_path)
    assert C.exam_sha256(rows) == lock["exam_content_sha256"], "content hash mismatch"
    harness_ok = all(_sha_file(f"/h/{f}") == h for f, h in lock["harness_sha256"].items() if f in FILES)
    sz = size_check.measure(art, mem, lock["base_fp16_bytes"], lock["budget_fraction"])
    view = C.candidate_view(rows); y = C.labels(rows)
    try:   # harness/infra failures before any scoring RELEASE the claim (no label was used); candidate crashes do not
        # --- base bar score (own process; stage 2 reuses stage 1 predictions) ---
        if prev: bp, bs = prev["_base_pred"], prev["base_seconds"]
        else:
            o = run_base.remote(view, lock["base_id"]); bp, bs = o["pred"], o["seconds"]
        # --- candidate: no-network sandbox, artifact+memory only ---
        inp_rel = f"inputs/{key.replace(':', '_')}.json"   # question rows only, no answers
        os.makedirs(os.path.dirname(f"/cand/{inp_rel}"), exist_ok=True); json.dump(view, open(f"/cand/{inp_rel}", "w")); candvol.commit()
        cmd = ["python", "/h/worker.py", "--artifact", f"/cand/{spec['artifact_vol']}", "--inputs", f"/cand/{inp_rel}"]
        if mem: cmd += ["--memory", f"/cand/{spec['memory_vol']}"]
        app_ = modal.App.lookup("tenpct-v2-exam", create_if_missing=True)
        sb = modal.Sandbox.create("sleep", "infinity", app=app_, image=cand_img, gpu=GPU, block_network=True, timeout=7200,
                                  volumes={"/cand": candvol.read_only()})
        try:
            p = sb.exec(*cmd); out = p.stdout.read(); err = p.stderr.read(); rc = p.wait()
        finally:
            sb.terminate()
    except Exception as e:
        import traceback
        os.makedirs("/exam/aborts", exist_ok=True)
        json.dump(dict(key=key, t=C.now(), labels_scored=False, error=traceback.format_exc()[-3000:]), open(f"/exam/aborts/{key.replace(':', '_')}_{int(time.time())}.json", "w"))
        claims.pop(key); import shutil; shutil.rmtree(rdir, ignore_errors=True); examvol.commit()
        print("INFRA_ABORT (claim released, nothing scored):", repr(e)); raise
    line = [l for l in out.splitlines() if l.startswith("CAND_JSON:")]
    if rc != 0 or not line:
        rec.update(status="candidate_crashed", returncode=rc, stderr_tail=err[-2000:], size=sz, finished_utc=C.now())
        json.dump(rec, open(f"{rdir}/receipt.json", "w")); examvol.commit()
        print("RECEIPT_JSON:" + json.dumps(rec)); return rec
    o = json.loads(line[0][len("CAND_JSON:"):]); cp = o["pred"]
    bad = [c for c in cp if c not in C.LETTERS]
    if bad: cp = [c if c in C.LETTERS else "?" for c in cp]
    rep = S.paired_report(y, bp, cp, margin=lock["margin"], alpha=lock["alpha"])
    valid = bool(sz["within_budget"] and harness_ok and not lock["standin"] and o["evidence"]["network"].startswith("blocked"))
    rec.update(status="completed", finished_utc=C.now(), standin=lock["standin"], size=sz, report=rep, harness_unchanged_since_lock=harness_ok,
               sandbox_evidence=o["evidence"], base_seconds=bs, cand_seconds=o["seconds"], base_id=lock["base_id"],
               lock_exam_file_sha256=lock["exam_file_sha256"], size_ok=bool(sz["within_budget"]),
               PASS=bool(sz["within_budget"] and rep["non_inferior"] and harness_ok and o["evidence"]["network"].startswith("blocked")),
               REAL_RESULT=valid, _base_pred=bp, cand_pred_sha256=C.sha256_bytes(json.dumps(cp).encode()), _cand_pred=cp)
    if prev: rec["stage1_report"] = prev["report"]; rec["PASS_BOTH_STAGES"] = bool(rec["PASS"] and prev["PASS"])
    json.dump(rec, open(f"{rdir}/receipt.json", "w")); examvol.commit()
    print("RECEIPT_JSON:" + json.dumps(rec))
    return rec


@app.function(image=orch_img, timeout=3600, volumes={"/cand": candvol})
def make_standin_fullcopy(base_id: str, dest: str):
    """Stand-in artifact: self-contained fp16 copy of a small base + predict.py (writes straight into the cand volume)."""
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "huggingface_hub"], check=True)
    from huggingface_hub import snapshot_download
    snapshot_download(base_id, local_dir=f"/cand/{dest}", allow_patterns=["*.safetensors", "*.json", "merges.txt", "vocab.json"])
    open(f"/cand/{dest}/predict.py", "w").write(open("/h/standin_predict.py").read())
    candvol.commit(); return os.listdir(f"/cand/{dest}")


@app.local_entrypoint()
def main(lock: str, tag: str, stage: int = 1, exam_file: str = "", artifact: str = "", artifact_vol: str = "",
         memory: str = "", memory_vol: str = "", receipts_dir: str = ""):
    L = json.load(open(lock))
    receipts_dir = receipts_dir or f"{HERE}/receipts/{'standin' if L['standin'] else 'REAL'}_{tag}"
    os.makedirs(receipts_dir, exist_ok=True)
    claim = f"{receipts_dir}/stage{stage}.claim"
    try:
        fd = os.open(claim, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o444)
    except FileExistsError:
        sys.exit(f"REFUSING: {claim} exists; the exam is single-use per stage.")
    os.write(fd, json.dumps(dict(t=time.time(), lock=lock, artifact=artifact or artifact_vol, memory=memory or memory_vol)).encode()); os.close(fd)
    if exam_file:   # exam (with labels) -> exam volume only
        with examvol.batch_upload(force=True) as b:
            b.put_file(exam_file, f"exams/{L['exam_file_sha256']}.jsonl")
    base = f"runs/{tag}/stage{stage}"
    if artifact:
        with candvol.batch_upload(force=True) as b:
            b.put_directory(artifact, f"{base}/artifact"); artifact_vol = f"{base}/artifact"
    if memory:
        with candvol.batch_upload(force=True) as b:
            b.put_file(memory, f"{base}/memory/{os.path.basename(memory)}"); memory_vol = f"{base}/memory/{os.path.basename(memory)}"
    spec = dict(lock=L, stage=stage, tag=tag, artifact_vol=artifact_vol, memory_vol=memory_vol or None)
    rec = run_exam.remote(spec)
    json.dump(rec, open(f"{receipts_dir}/stage{stage}.json", "w"), indent=2)
    print({k: rec.get(k) for k in ("status", "reason", "PASS", "REAL_RESULT", "size_ok")}, rec.get("report", {}).get("diff_cand_minus_base"))
