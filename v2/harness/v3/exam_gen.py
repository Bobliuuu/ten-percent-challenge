#!/usr/bin/env python
"""Single-use GENERATION exam runner (v3). Output = the exact input format of ../scorecard.py `score()` (exam part) + a hashed receipt.
  python exam_gen.py --exam EXAM.jsonl --artifact ART --base-id Qwen/Qwen3-0.6B --base-arch ART.base_arch.json --tag NAME --stage 1 \
                     --out-dir RUN/ [--budget-fraction 0.10] [--source-map S3=S2] [--isolation auto|netns|guard]
Exam rows (labels only in the orchestrator): {id, kind, source, query, tools, gold, [gold_alts]}.  Candidate and base get {id, query, tools} ONLY.
Single use: O_EXCL claim file receipts/<tag>/stage<N>.claim BEFORE any label is read; a crashed candidate consumes the claim (an infra failure
before scoring releases it). The locked exam is refused unless --allow-locked and TENPCT_FINAL=1."""
import argparse, json, os, secrets, sys, tempfile, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common3 as C, checks, runner, sandbox

HARNESS_FILES = ["common3.py", "checks.py", "sandbox.py", "runner.py", "exam_gen.py", "feedback_protocol.py", "final_proof.py", "gen_common.py", "base_gen.py", "netguard/sitecustomize.py"]


def harness_hashes(): return {f: C.sha_file(os.path.join(C.HERE, f)) for f in HARNESS_FILES if os.path.exists(os.path.join(C.HERE, f))}


def claim(receipts_dir, tag, stage, info):
    os.makedirs(os.path.join(receipts_dir, tag), exist_ok=True); p = os.path.join(receipts_dir, tag, f"stage{stage}.claim")
    try: fd = os.open(p, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o444)
    except FileExistsError: raise SystemExit(f"REFUSING: {p} exists; the exam is single-use per (tag, stage).")
    os.write(fd, json.dumps(info).encode()); os.close(fd); return p


def release(p):
    os.chmod(p, 0o644); os.remove(p)


def load_arch(path):
    d = json.load(open(path)); return d["arch"], d.get("config"), d["base_fp16_bytes"]


def evaluate(exam_rows, artifact, base_id, base_texts=None, workdir=None, isolation="auto", hide_dirs=(), python=sys.executable, base_cmd=None, canary=None,
             check_leak=True, max_new_tokens=512):
    """Core: returns dict with base/cand texts+correct arrays or a failure status. exam_rows carry labels; only the view leaves this process."""
    view = C.candidate_view(exam_rows); out = {}
    if check_leak:
        out["leak_view"] = checks.leak_scan(view, exam_rows, canary)
        if not out["leak_view"]["ok"]: out["status"] = "leak_in_view"; return out
    with tempfile.TemporaryDirectory(dir=workdir) as td:
        if base_texts is None:
            b = runner.run_base(base_id, view, td, python=python, base_cmd=base_cmd, max_new_tokens=max_new_tokens); base_texts = b["texts"]; out["base_seconds"] = b["seconds"]; out["base_out_sha256"] = b["out_sha256"]
        c = runner.run_candidate(artifact, view, td, mode=isolation, hide_dirs=hide_dirs, python=python)
    out["network_attempts"] = c["evidence"].get("network_attempts_logged", [])
    out.update(status=c["status"], cand_evidence=c["evidence"], cand_seconds=c["seconds"], artifact_unchanged_by_answer=c["artifact_unchanged"], cand_stderr_tail=c["stderr_tail"])
    if c["status"] != "completed": return out
    out["cand_out_sha256"] = c["out_sha256"]
    if check_leak and c["prompts"] is not None:
        out["leak_prompts"] = checks.prompt_leak_scan(c["prompts"], exam_rows, canary)
    if canary: out["leak_outputs"] = checks.scan_text_for_canary(c["texts"], canary)
    sb, sc = runner.score(exam_rows, base_texts), runner.score(exam_rows, c["texts"])
    out.update(base_texts=base_texts, cand_texts=c["texts"], base_correct=sb["correct"], cand_correct=sc["correct"], base_parse_fail=sb["parse_fail"], cand_parse_fail=sc["parse_fail"],
               agree=runner.agree(base_texts, c["texts"]), kind=[r["kind"] for r in exam_rows], source_raw=[r["source"] for r in exam_rows])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exam", required=True); ap.add_argument("--artifact", required=True); ap.add_argument("--base-id", required=True); ap.add_argument("--base-arch", required=True)
    ap.add_argument("--tag", required=True); ap.add_argument("--stage", type=int, default=1); ap.add_argument("--out-dir", required=True)
    ap.add_argument("--budget-fraction", type=float, default=0.10); ap.add_argument("--source-map", default=""); ap.add_argument("--isolation", default="auto")
    ap.add_argument("--allow-locked", action="store_true"); ap.add_argument("--python", default=sys.executable); ap.add_argument("--receipts-dir")
    ap.add_argument("--base-texts", help="reuse base outputs (JSON list of {id,text}) e.g. for stage 2"); ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--limit", type=int, help="dev only: first N rows"); ap.add_argument("--base-cmd-json", help="tests only: argv template for a stub base, with {in} {out}")
    a = ap.parse_args()
    is_locked = C.locked_guard(a.exam, a.allow_locked)
    os.makedirs(a.out_dir, exist_ok=True); rdir = a.receipts_dir or os.path.join(a.out_dir, "receipts")
    arch, bcfg, fp16 = load_arch(a.base_arch); art_sha0 = C.tree_sha256(a.artifact)
    cl = claim(rdir, a.tag, a.stage, dict(t=C.now(), exam_sha256=C.sha_file(a.exam), artifact_tree_sha256=art_sha0, locked=is_locked))
    try:
        rows = C.load_jsonl(a.exam)
        if a.limit: rows = rows[:a.limit]
        canary = "CANARY-" + secrets.token_hex(8)                      # lives only on the label side; must never appear candidate-side
        labeled = [dict(r, _canary=canary) for r in rows]
        smap = dict(kv.split("=") for kv in a.source_map.split(",") if kv)
        size = checks.size_check(a.artifact, fp16, a.budget_fraction); nnp = checks.no_new_parts(a.artifact, arch, bcfg)
        label_dirs = [os.path.dirname(os.path.abspath(a.exam))]
        bt = None
        if a.base_texts: bt = [o["text"] for o in json.load(open(a.base_texts))]
        r = evaluate(labeled, a.artifact, a.base_id, base_texts=bt, isolation=a.isolation, hide_dirs=label_dirs, python=a.python, canary=canary, max_new_tokens=a.max_new_tokens, base_cmd=json.loads(a.base_cmd_json) if a.base_cmd_json else None)
    except BaseException:
        release(cl); raise                                              # infra failure before any scoring: claim released
    rec = dict(status=r["status"], tag=a.tag, stage=a.stage, utc=C.now(), exam_file_sha256=C.sha_file(a.exam), n_rows=len(rows), exam_rows_sha256=C.rows_sha256(rows), locked_exam=is_locked,
               base_id=a.base_id, base_arch_sha256=C.sha_file(a.base_arch), artifact_tree_sha256_before=art_sha0, artifact_tree_sha256_after=C.tree_sha256(a.artifact), scorer_sha256=C.scorer_sha(),
               harness_sha256=harness_hashes(), size=size, no_new_parts=nnp, checks={k: r.get(k) for k in ("leak_view", "leak_prompts", "leak_outputs")},
               cand_evidence=r.get("cand_evidence"), network_attempts=r.get("network_attempts"), artifact_unchanged_by_answer=r.get("artifact_unchanged_by_answer"), base_out_sha256=r.get("base_out_sha256"), cand_out_sha256=r.get("cand_out_sha256"))
    if r["status"] == "leak_in_view": release(cl)      # nothing was run or scored: the claim is not consumed
    if r["status"] != "completed":
        rec["stderr_tail"] = r.get("cand_stderr_tail"); C.write_json(os.path.join(a.out_dir, f"{a.tag}.stage{a.stage}.receipt.json"), rec); print("RECEIPT", rec["status"]); sys.exit(2)
    ev = r["cand_evidence"]; rec["isolation_ok"] = bool(ev["network"].startswith("blocked") and ev["mode"] == "netns")
    res = dict(base_correct=r["base_correct"], cand_correct=r["cand_correct"], kind=r["kind"], source=[smap.get(s, s) for s in r["source_raw"]], agree=r["agree"],
               ids=[x["id"] for x in rows], base_texts=r["base_texts"], cand_texts=r["cand_texts"])
    rec["accuracy"] = dict(base=sum(res["base_correct"]) / len(rows), cand=sum(res["cand_correct"]) / len(rows), agree=sum(res["agree"]) / len(rows),
                           base_parse_fail=sum(r["base_parse_fail"]), cand_parse_fail=sum(r["cand_parse_fail"]))
    rec["all_checks_ok"] = bool(size["ok"] and nnp["ok"] and r["leak_view"]["ok"] and r.get("leak_prompts", {"ok": True})["ok"] and r["leak_outputs"]["ok"] and r["artifact_unchanged_by_answer"] and not r["network_attempts"])
    rec["REAL_RESULT"] = bool(rec["all_checks_ok"] and rec["isolation_ok"] and is_locked)
    C.write_json(os.path.join(a.out_dir, f"{a.tag}.stage{a.stage}.exam_arrays.json"), res)
    C.write_json(os.path.join(a.out_dir, f"{a.tag}.stage{a.stage}.receipt.json"), rec, ro=True)
    print("RECEIPT_JSON:" + json.dumps({k: rec[k] for k in ("status", "accuracy", "all_checks_ok", "isolation_ok", "REAL_RESULT")}))


if __name__ == "__main__": main()
