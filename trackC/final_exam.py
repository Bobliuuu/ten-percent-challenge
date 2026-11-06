#!/usr/bin/env python
"""Final exam on the locked split. ONE run per stage, enforced by an O_EXCL receipt claimed BEFORE any labels are read.
  stage 1: final_exam.py --artifact DIR [--memory MEM0]
  stage 2: final_exam.py --artifact DIR --memory MEM1 --stage 2      (post-feedback re-run, rule 5)
Test stand-in: add --split dev --lock <dev LOCK.json> --receipt-dir <scratch dir>.
Artifact contract: DIR/predict.py defines load(artifact_dir, memory_path_or_None) -> f(list[str]) -> list[int]."""
import argparse, datetime, json, os, shutil, subprocess, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
import size_check

ap = argparse.ArgumentParser()
ap.add_argument("--artifact", required=True); ap.add_argument("--memory")
ap.add_argument("--stage", type=int, choices=(1, 2), default=1)
ap.add_argument("--split", default="validation")
ap.add_argument("--lock", default=os.path.join(C.HERE, "LOCK.json"))
ap.add_argument("--receipt-dir", default=os.path.join(C.HERE, "receipts"))
ap.add_argument("--no-sandbox", action="store_true", help="debug only; recorded in receipt")
a = ap.parse_args()

default_rd = os.path.join(C.HERE, "receipts")
real = a.split == "validation"
if not real and os.path.abspath(a.receipt_dir) == default_rd:
    sys.exit("stand-in runs must use a scratch --receipt-dir, never the real receipts dir")
lock = json.load(open(a.lock))
if lock["split"] != a.split:
    sys.exit(f"lock is for split {lock['split']!r}, not {a.split!r}")
os.makedirs(a.receipt_dir, exist_ok=True)
rpath = os.path.join(a.receipt_dir, f"receipt_stage{a.stage}.json")
art = os.path.abspath(a.artifact)
if not os.path.isfile(os.path.join(art, "predict.py")):
    sys.exit("artifact must be a directory containing predict.py")
mem = os.path.abspath(a.memory) if a.memory else None
if mem and not os.path.isfile(mem):
    sys.exit("memory file not found")

# ---- pre-flight that does not touch the exam ----
art_sha, mem_sha = C.tree_sha256(art), (C.sha256_file(mem) if mem else None)
prev = None
if a.stage == 2:
    p1 = os.path.join(a.receipt_dir, "receipt_stage1.json")
    if not os.path.exists(p1): sys.exit("stage 2 requires a completed stage-1 receipt")
    prev = json.load(open(p1))
    if prev.get("status") != "completed": sys.exit("stage-1 receipt not completed")
    if prev["artifact_tree_sha256"] != art_sha: sys.exit("REFUSING: artifact (weights/code) changed since stage 1; only memory may change")
    if prev["memory_sha256"] == mem_sha: sys.exit("REFUSING: memory identical to stage 1; nothing was fed back")
sz = size_check.measure(art, mem)
if not sz["within_10pct"]:
    print("WARNING: size gate FAILED; exam will still be run once and the receipt marks it INVALID")

# ---- claim the run (refuse a second one, even after a crash) ----
claim = dict(status="started", stage=a.stage, started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
             artifact_tree_sha256=art_sha, memory_sha256=mem_sha)
try:
    fd = os.open(rpath, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
except FileExistsError:
    sys.exit(f"REFUSING: {rpath} exists. The exam is single-use per stage (archive nothing; this is a protocol breach if re-run).")
with os.fdopen(fd, "w") as f: json.dump(claim, f)

# ---- verify lock, load exam ----
rows, _ = C.load_split(a.split)
cur = C.rows_sha256(rows)
lock_ok = cur == lock["content_sha256"] and len(rows) == lock["n_rows"]
if not lock_ok:
    claim.update(status="aborted_lock_mismatch"); C.write_json_atomic(rpath, claim)
    sys.exit("exam content does not match LOCK.json")
y = [r[1] for r in rows]; texts = [r[2] for r in rows]
harness_changed = C.harness_hashes() != lock["harness_sha256"]

tmp = os.path.realpath(tempfile.mkdtemp(prefix="exam_"))
json.dump(texts, open(f"{tmp}/texts.json", "w"))
py = sys.executable
worker = os.path.join(C.HERE, "_worker.py")


def run_base():
    if prev is not None:
        return prev["_base_pred"], prev["base_seconds"]
    subprocess.run([py, worker, "--mode", "base", "--texts", f"{tmp}/texts.json", "--out", f"{tmp}/base.json"], check=True)
    o = json.load(open(f"{tmp}/base.json")); return o["pred"], o["seconds"]


def run_cand():
    # copy artifact+memory into a scratch dir; run with network off and the HF cache unreadable (macOS sandbox-exec)
    ad = f"{tmp}/artifact"; shutil.copytree(art, ad)
    assert C.tree_sha256(ad) == art_sha
    cmd = [py, worker, "--mode", "artifact", "--artifact", ad, "--texts", f"{tmp}/texts.json", "--out", f"{tmp}/cand.json"]
    if mem:
        os.makedirs(f"{tmp}/mem"); shutil.copy(mem, f"{tmp}/mem/{os.path.basename(mem)}"); cmd += ["--memory", f"{tmp}/mem/{os.path.basename(mem)}"]
    env = dict(os.environ, HOME=tmp, TMPDIR=tmp, HF_HOME=f"{tmp}/hf", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_DATASETS_OFFLINE="1")
    sandboxed = False
    if not a.no_sandbox and shutil.which("sandbox-exec"):
        hf = os.path.realpath(os.path.expanduser("~/.cache/huggingface"))
        prof = (f'(version 1)(allow default)(deny network*)(deny file-read* (subpath "{hf}"))'
                f'(deny file-write*)(allow file-write* (subpath "{tmp}") (subpath "/dev"))')
        cmd = ["sandbox-exec", "-p", prof] + cmd; sandboxed = True
    r = subprocess.run(cmd, env=env)
    if r.returncode != 0:
        claim.update(status="candidate_crashed", returncode=r.returncode, sandboxed=sandboxed)
        C.write_json_atomic(rpath, claim, ro=True)
        sys.exit("candidate crashed; run is CONSUMED (receipt written). Fix is a protocol breach to re-run.")
    o = json.load(open(f"{tmp}/cand.json")); return o["pred"], o["seconds"], sandboxed


try:
    bp, bs = run_base()
    cp, cs, sandboxed = run_cand()
    rep = C.paired_report(y, bp, cp)
    valid = bool(sz["within_10pct"]) and lock_ok
    rec = dict(claim, status="completed", finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
               split=a.split, standin=not real, lock_content_sha256=lock["content_sha256"], lock_utc=lock["locked_utc"],
               exam_rows_sha256=cur, harness_changed_since_lock=harness_changed, sandboxed_candidate=sandboxed,
               size=sz, report=rep, base_seconds=bs, cand_seconds=cs,
               cand_pred_sha=C.hashlib.sha256(json.dumps(cp).encode()).hexdigest(),
               valid_size=bool(sz["within_10pct"]),
               PASS=bool(valid and rep["non_inferior"]))
    if a.stage == 2:
        rec["stage1_report"] = prev["report"]
        rec["stage1_cand_acc"] = prev["report"]["cand_acc"]
        rec["PASS_BOTH_STAGES"] = bool(rec["PASS"] and prev["PASS"])
    rec["_base_pred"] = bp
    C.write_json_atomic(rpath, rec, ro=True)
finally:
    shutil.rmtree(tmp, ignore_errors=True)

r = rep
print(f"stage {a.stage} on {a.split} (n={r['n']})  size={100*sz['byte_ratio']:.2f}% of base  sandboxed={sandboxed}  harness_changed={harness_changed}")
print(f"base acc {100*r['base_acc']:.2f}% {tuple(round(100*x,2) for x in r['base_ci95'])} | cand acc {100*r['cand_acc']:.2f}% {tuple(round(100*x,2) for x in r['cand_ci95'])}")
print(f"discordant: base-only-right b={r['base_only_right_b']}  cand-only-right c={r['cand_only_right_c']}  McNemar exact p={r['mcnemar_exact_two_sided_p']:.4g}")
print(f"diff {100*r['diff_cand_minus_base']:+.2f} pt; exact 95% one-sided lower bound {100*r['exact_one_sided_lower_bound_on_diff']:+.2f} pt; "
      f"NI(margin 1pt) p={r['ni_exact_unconditional_p']:.4g} -> {'NON-INFERIOR' if r['non_inferior'] else 'NOT shown non-inferior'}")
print("PASS" if rec["PASS"] else "FAIL", f"(size {'ok' if sz['within_10pct'] else 'OVER'})", f"receipt: {rpath}")
