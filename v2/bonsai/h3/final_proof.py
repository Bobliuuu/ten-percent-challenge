#!/usr/bin/env python
"""One-go final proof: exam stage 1 -> feedback protocol (locks, real/shuffled/empty) -> exam stage 2 (after feedback) -> scorecard.py input + verdict.
  python final_proof.py --exam EXAM.jsonl --artifact ART --base-id ID --base-arch ART.base_arch.json --annotations ann.jsonl --teach teach.jsonl --neighbor nb.jsonl \
        --tag NAME --out-dir RUN/ [--source-map S3=S2] [--budget-fraction 0.10] [--allow-locked (needs env TENPCT_FINAL=1)]
Never edits the artifact (feedback runs on copies). Writes RUN/<tag>.scorecard_input.json, RUN/<tag>.scorecard.json, RUN/<tag>.PROOF_SUMMARY.json."""
import argparse, importlib.util, json, os, secrets, subprocess, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common3 as C, checks, exam_gen, feedback_protocol as FP


def load_scorecard():
    spec = importlib.util.spec_from_file_location("scorecard", os.path.join(C.V2, "harness", "scorecard.py")); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


def nnp_for_scorecard(before, after):
    extra = set()
    for n in (before, after):
        extra |= set(n["extra_files"]) | {"tensor+" + t for t in n["extra_tensors"]} | {"tensor-" + t for t in n["missing_tensors"]} | {"shape:" + t for t in n["shape_mismatch"]}
        extra |= {"dtype:" + t for t in n["dtype_mismatch"]} | {"config:" + k for k in n["config_arch_diff"]} | {"net:" + l for l in n["network_lines"]} | {"constrained:" + l for l in n["constrained_lines"]}
    return dict(extra_files=sorted(extra), grep_train_hits=max(before["grep_train_hits"], after["grep_train_hits"]))


def needles_from(rows, limit=300, min_len=12):
    """verbatim strings that must not be inside the artifact: gold calls (name+arguments JSON), queries of annotations / teach / neighbor / exam"""
    import json as _j
    ns = set()
    for r in rows:
        for g in r.get("gold", []):
            if g.get("arguments"): ns.add(_j.dumps(dict(name=g["name"], arguments=g["arguments"]), ensure_ascii=False))
        ns.add(r["query"])
    return sorted((n for n in ns if len(n) >= min_len), key=lambda s: (-len(s), s))[:limit]


def allowed_manifest_for(artifact):
    """files of the artifact tree that match the harness allow-list (checks.ALLOWED_PATTERNS, no subdirs); anything else present is an extra file. Weights are not sha-pinned (feedback edits them)."""
    import fnmatch
    return {rel: None for rel, _ in C.tree_files(artifact) if "/" not in rel and any(fnmatch.fnmatch(rel, p) for p in checks.ALLOWED_PATTERNS)}


def base_runs_for(rows, base_id, base_cmd, python, first_texts, first_bs=8, extra_bs=(4, 16)):
    """K>=3 original-model runs differing in batch size (noise floor): the stage-1 run (bs 8) + extra runs, each in its own process; scored with the T1 scorer."""
    import tempfile, runner
    view = C.candidate_view(rows); runs = [dict(correct=runner.score(rows, first_texts)["correct"], engine="hf", batch=first_bs)]
    for bs in extra_bs:
        with tempfile.TemporaryDirectory() as td: t = runner.run_base(base_id, view, td, python=python, base_cmd=base_cmd, bs=bs)["texts"]
        runs.append(dict(correct=runner.score(rows, t)["correct"], engine="hf", batch=bs))
    return runs


def assemble(exam_arrays, fb_block, size_before, nnp_before, nnp_after, fp16, generative=True, base_runs=None, artifact_tree=None):
    return dict(base_correct=exam_arrays["base_correct"], **({"base_runs": base_runs} if base_runs else {}), **({"artifact_tree": artifact_tree} if artifact_tree else {}), cand_correct=exam_arrays["cand_correct"], kind=exam_arrays["kind"], source=exam_arrays["source"], agree=exam_arrays["agree"],
                size=dict(cand_bytes=size_before["candidate_bytes"], base_fp16_bytes=fp16), base_params=fp16 // 2, generative=generative,
                kinds_real_output=bool(generative and nnp_before["constrained_decoding_hits"] == 0 and nnp_after["constrained_decoding_hits"] == 0), chance=float(os.environ.get("LABEL_CHANCE","0.0")),
                no_new_parts=nnp_for_scorecard(nnp_before, nnp_after), feedback=fb_block)


def main():
    ap = argparse.ArgumentParser()
    for k in ("exam", "artifact", "base-id", "base-arch", "annotations", "teach", "neighbor", "tag", "out-dir"): ap.add_argument("--" + k, required=True)
    ap.add_argument("--budget-fraction", type=float, default=0.10); ap.add_argument("--source-map", default=""); ap.add_argument("--isolation", default="auto"); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--allow-locked", action="store_true"); ap.add_argument("--no-base-runs", dest="base_runs", action="store_false"); ap.add_argument("--python", default=sys.executable); ap.add_argument("--limit", type=int); ap.add_argument("--base-cmd-json")
    a = ap.parse_args(); os.makedirs(a.out_dir, exist_ok=True)
    C.locked_guard(a.exam, a.allow_locked)
    me = os.path.join(C.HERE, "exam_gen.py")
    cmd = [a.python, me, "--exam", a.exam, "--artifact", a.artifact, "--base-id", a.base_id, "--base-arch", a.base_arch, "--tag", a.tag, "--stage", "1", "--out-dir", a.out_dir,
           "--budget-fraction", str(a.budget_fraction), "--source-map", a.source_map, "--isolation", a.isolation, "--python", a.python] + (["--allow-locked"] if a.allow_locked else []) + (["--limit", str(a.limit)] if a.limit else []) + (["--base-cmd-json", a.base_cmd_json] if a.base_cmd_json else [])
    r = subprocess.run(cmd); 
    if r.returncode: sys.exit("stage 1 failed; see receipt in " + a.out_dir)
    ex1 = json.load(open(os.path.join(a.out_dir, f"{a.tag}.stage1.exam_arrays.json"))); rc1 = json.load(open(os.path.join(a.out_dir, f"{a.tag}.stage1.receipt.json")))
    rows = C.load_jsonl(a.exam)[:a.limit] if a.limit else C.load_jsonl(a.exam)
    arch, bcfg, fp16 = exam_gen.load_arch(a.base_arch)
    # stage 2 = the exam AFTER feedback on the real arm: single-use claim of its own
    cl = exam_gen.claim(os.path.join(a.out_dir, "receipts"), a.tag, 2, dict(t=C.now(), exam_sha256=rc1["exam_file_sha256"], artifact_tree_sha256=rc1["artifact_tree_sha256_before"]))
    try:
        fb = FP.run_protocol(a.artifact, a.base_id, a.base_arch, dict(ann=a.annotations, teach=a.teach, nb=a.neighbor), os.path.join(a.out_dir, "feedback"), a.budget_fraction, a.seed, a.isolation,
                             python=a.python, base_cmd=json.loads(a.base_cmd_json) if a.base_cmd_json else None, exam_rows=rows, exam_base_texts=ex1["base_texts"], canary="CANARY-" + secrets.token_hex(8), hide_dirs=[os.path.dirname(os.path.abspath(a.exam))])
    except BaseException: exam_gen.release(cl); raise
    if fb.get("status") != "completed": sys.exit("feedback protocol did not complete: " + json.dumps({k: v.get("status") for k, v in fb["arms"].items()}))
    nnp_before = rc1["no_new_parts"]; nnp_after = fb["arms"]["real"]["no_new_parts_after"]
    bruns = base_runs_for(rows, a.base_id, json.loads(a.base_cmd_json) if a.base_cmd_json else None, a.python, ex1["base_texts"]) if a.base_runs else None
    ann_rows, teach_rows, nb_rows = (C.load_jsonl(p) for p in (a.annotations, a.teach, a.neighbor))
    tree = dict(dir=os.path.abspath(a.artifact), allowed_manifest=allowed_manifest_for(a.artifact), needles=needles_from(ann_rows + teach_rows + nb_rows + rows))
    inp = assemble(ex1, fb["feedback_block"], rc1["size"], nnp_before, nnp_after, fp16, base_runs=bruns, artifact_tree=tree)
    C.write_json(os.path.join(a.out_dir, f"{a.tag}.scorecard_input.json"), inp)
    SC = load_scorecard(); o = SC.score(inp); SC.show(o)
    C.write_json(os.path.join(a.out_dir, f"{a.tag}.scorecard.json"), o)
    summ = dict(tag=a.tag, utc=C.now(), verdict=o["verdict"], failed_gates=o["failed_gates"], maximin=o["maximin"], exam_receipt_sha256=C.sha_file(os.path.join(a.out_dir, f"{a.tag}.stage1.receipt.json")),
                feedback_lock_sha256=fb["lock_sha256"], empty_byte_identical=fb["empty_byte_identical"], teach_acc=dict(base=sum(fb["feedback_block"]["teach"]["base_correct"]) / fb["feedback_block"]["teach"]["n"],
                real=sum(fb["feedback_block"]["teach"]["real_correct"]) / fb["feedback_block"]["teach"]["n"], mixed=sum(fb["feedback_block"]["teach"]["mixed_label_correct"]) / fb["feedback_block"]["teach"]["n"], wrong_tool=sum(fb["feedback_block"]["teach"]["random_wrong_tool_correct"]) / fb["feedback_block"]["teach"]["n"], shuffled=sum(fb["feedback_block"]["teach"]["shuffled_correct"]) / fb["feedback_block"]["teach"]["n"],
                empty=sum(fb["feedback_block"]["teach"]["empty_correct"]) / fb["feedback_block"]["teach"]["n"]), checks_ok_stage1=rc1["all_checks_ok"], isolation_ok=rc1["isolation_ok"], REAL_RESULT=rc1["REAL_RESULT"] and fb["empty_byte_identical"],
                feedback_arms={k: {kk: v[kk] for kk in ("rc", "changed_files", "bytes_added", "byte_identical")} for k, v in fb["arms"].items()}, harness_sha256=exam_gen.harness_hashes())
    C.write_json(os.path.join(a.out_dir, f"{a.tag}.PROOF_SUMMARY.json"), summ, ro=True); print(json.dumps(summ["teach_acc"]), "empty_byte_identical", fb["empty_byte_identical"])


if __name__ == "__main__": main()
