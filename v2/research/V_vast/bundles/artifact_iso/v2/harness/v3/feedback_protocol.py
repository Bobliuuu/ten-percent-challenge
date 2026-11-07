#!/usr/bin/env python
"""Feedback protocol runner. Locks the teach-set and neighbor-set (hashes) BEFORE any feedback, then on three FRESH COPIES of the artifact runs the
`feedback` command with (real | shuffled-label | empty) annotations, checks size / no-new-parts / byte-identity afterwards and evaluates all three
(teach-set for all, neighbor + exam for the real one). Writes the `feedback` block of scorecard.py's input.
  python feedback_protocol.py --artifact ART --base-id ID --base-arch A.json --annotations ann.jsonl --teach teach.jsonl --neighbor nb.jsonl \
       --out-dir RUN/ [--exam EXAM.jsonl --exam-base-texts exam_arrays.json] [--budget-fraction 0.10] [--seed 0]
Row format for all three files: {id, kind, source, query, tools, gold:[{name,arguments}]}  (annotations: the corrected call(s) = `gold`)."""
import argparse, copy, json, os, random, shutil, subprocess, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common3 as C, checks, runner, sandbox, exam_gen

MIN_TEACH, MAX_ANN = 25, 25


def gold_key(r): return json.dumps(r.get("gold"), sort_keys=True)
def item_key(r): return C.sha_bytes(json.dumps([r["query"], r["tools"]], sort_keys=True).encode())


def validate_sets(ann, teach, nb, exam_rows=()):
    """-> list of problems (empty = ok). Disjoint ids AND disjoint (query,tools) content across annotations / teach / neighbor / exam."""
    p = []
    if not (1 <= len(ann) <= MAX_ANN): p.append(f"annotations must be 1..{MAX_ANN}, got {len(ann)}")
    if len(teach) < MIN_TEACH: p.append(f"teach-set must be >= {MIN_TEACH}, got {len(teach)}")
    if not nb: p.append("neighbor-set empty")
    sets = dict(ann=ann, teach=teach, neighbor=nb, exam=list(exam_rows))
    for n, s in sets.items():
        if len({r["id"] for r in s}) != len(s): p.append(f"{n}: duplicate ids")
    names = list(sets)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = sets[names[i]], sets[names[j]]
            if {r["id"] for r in a} & {r["id"] for r in b}: p.append(f"{names[i]} and {names[j]} share ids")
            if {item_key(r) for r in a} & {item_key(r) for r in b}: p.append(f"{names[i]} and {names[j]} share (query,tools) content")
    for n in ("ann", "teach"):
        if any(not r.get("gold") for r in sets[n]): p.append(f"{n}: rows without gold")
    return p


def shuffled_annotations(ann, seed=0):
    """Permute the corrected labels across annotations so that no item keeps its own (derangement when possible). Same prompts, wrong labels."""
    keys = [gold_key(a) for a in ann]
    if len(set(keys)) < 2: return None, "degenerate: all annotations have the same gold; shuffling is impossible"
    rng = random.Random(seed)
    for _ in range(1000):
        perm = list(range(len(ann))); rng.shuffle(perm)
        if all(keys[perm[i]] != keys[i] for i in range(len(ann))): break
    else: return None, "no derangement found"
    return [dict(a, gold=copy.deepcopy(ann[perm[i]]["gold"])) for i, a in enumerate(ann)], None


def wrong_tool_annotations(ann, seed=0):
    """random-wrong-tool control: same prompts, each corrected call replaced by a call to a DIFFERENT tool (random pick from the item's own tool list, else from
    the other annotations' tools), arguments kept."""
    rng = random.Random(seed + 17); pool = sorted({t["name"] for a in ann for t in a["tools"]}); out = []
    for a in ann:
        names = [t["name"] for t in a["tools"]]; g0 = a["gold"][0]["name"]; cand = sorted(n for n in names if n != g0) or sorted(n for n in pool if n != g0)
        if not cand: return None, "no alternative tool name available"
        w = rng.choice(cand); out.append(dict(a, gold=[dict(name=w, arguments=copy.deepcopy(g["arguments"])) for g in a["gold"]]))
    return out, None


def mixed_annotations(ann, seed=0):
    """50/50 mixed-label control: first half of a seeded permutation keeps the TRUE label, the other half gets a wrong-tool label."""
    rng = random.Random(seed + 29); idx = list(range(len(ann))); rng.shuffle(idx); half = set(idx[:len(ann) // 2])
    wt, why = wrong_tool_annotations(ann, seed + 1)
    if wt is None: return None, why
    return [wt[i] if i in half else ann[i] for i in range(len(ann))], None


def family(r):
    g = r.get("gold") or []
    return g[0]["name"] if g else "none"


def write_lock(path, files, ann, teach, nb, art_sha, extra):
    lock = dict(utc=C.now(), annotations=dict(n=len(ann), file_sha256=C.sha_file(files["ann"]), ids=[r["id"] for r in ann]),
                teach=dict(n=len(teach), file_sha256=C.sha_file(files["teach"]), ids=[r["id"] for r in teach]),
                neighbor=dict(n=len(nb), file_sha256=C.sha_file(files["nb"]), ids=[r["id"] for r in nb]), artifact_tree_sha256=art_sha, scorer_sha256=C.scorer_sha(), **extra)
    if os.path.exists(path):
        old = json.load(open(path))
        for k in ("annotations", "teach", "neighbor"):
            if old[k]["file_sha256"] != lock[k]["file_sha256"]: raise SystemExit(f"REFUSING: feedback lock exists and {k} file changed since it was locked")
        if old["artifact_tree_sha256"] != art_sha: raise SystemExit("REFUSING: artifact changed since the feedback lock")
        return old
    C.write_json(path, lock, ro=True); return lock


def apply_feedback(artifact, anns, name, work, mode, hide_dirs, python, scale_ok_files=None):
    """Fresh copy -> run ART/feedback.py in the sandbox -> report. The copy dir is returned for evaluation."""
    dst = os.path.join(work, name); shutil.copytree(artifact, dst, symlinks=True)
    annp = os.path.join(work, f"{name}.annotations.jsonl"); C.dump_jsonl(anns, annp)
    before = C.tree_sha256(dst); files_before = {r: C.sha_file(a) for r, a in C.tree_files(dst)}
    r = sandbox.run([python, os.path.join(dst, "feedback.py"), "--artifact", dst, "--annotations", annp], mode=mode, hide_dirs=hide_dirs, python=python)
    after = C.tree_sha256(dst); files_after = {rl: C.sha_file(a) for rl, a in C.tree_files(dst)}
    changed = sorted(k for k in set(files_before) | set(files_after) if files_before.get(k) != files_after.get(k))
    return dst, dict(name=name, rc=r["rc"], stdout_tail=r["stdout"][-400:], stderr_tail=r["stderr"][-800:], evidence=r["evidence"], n_annotations=len(anns),
                     tree_sha256_before=before, tree_sha256_after=after, byte_identical=before == after, changed_files=changed,
                     added_files=sorted(set(files_after) - set(files_before)), removed_files=sorted(set(files_before) - set(files_after)))


def run_protocol(artifact, base_id, base_arch, files, out_dir, budget_fraction=0.10, seed=0, isolation="auto", python=sys.executable, base_cmd=None,
                 exam_rows=None, exam_base_texts=None, canary=None, hide_dirs=()):
    os.makedirs(out_dir, exist_ok=True)
    arch, bcfg, fp16 = exam_gen.load_arch(base_arch)
    ann, teach, nb = (C.load_jsonl(files[k]) for k in ("ann", "teach", "nb"))
    probs = validate_sets(ann, teach, nb, exam_rows or [])
    if probs: raise SystemExit("INVALID SETS:\n  " + "\n  ".join(probs))
    art_sha = C.tree_sha256(artifact)
    lock = write_lock(os.path.join(out_dir, "FEEDBACK_LOCK.json"), files, ann, teach, nb, art_sha, dict(seed=seed, base_id=base_id))   # <- before ANY feedback
    shuf, why = shuffled_annotations(ann, seed)
    if shuf is None: raise SystemExit("SHUFFLE CONTROL IMPOSSIBLE: " + why)
    wrong, why = wrong_tool_annotations(ann, seed)
    if wrong is None: raise SystemExit("WRONG-TOOL CONTROL IMPOSSIBLE: " + why)
    mixed, why = mixed_annotations(ann, seed)
    if mixed is None: raise SystemExit("MIXED CONTROL IMPOSSIBLE: " + why)
    size0 = checks.size_check(artifact, fp16, budget_fraction)
    evalrows = teach + nb; canary = canary or "CANARY-fb"
    labeled = [dict(r, _canary=canary) for r in evalrows]
    work = tempfile.mkdtemp(prefix="fbproto_"); lab_dirs = sorted(set(os.path.dirname(os.path.abspath(f)) for k, f in files.items() if k != "ann") | set(map(os.path.abspath, hide_dirs)))
    # base + candidate-before on teach+neighbor
    view = C.candidate_view(evalrows)
    with tempfile.TemporaryDirectory(dir=work) as td: bt = runner.run_base(base_id, view, td, python=python, base_cmd=base_cmd)["texts"]
    out = dict(lock_sha256=C.sha_file(os.path.join(out_dir, "FEEDBACK_LOCK.json")), shuffle_seed=seed, arms={})
    nt = len(teach)
    def ev(art): return exam_gen.evaluate(labeled, art, base_id, base_texts=bt, workdir=work, isolation=isolation, hide_dirs=lab_dirs, python=python, canary=canary)
    before = ev(artifact)
    if before["status"] != "completed": raise SystemExit("candidate-before failed: " + str(before.get("status")) + " " + str(before.get("cand_stderr_tail")))
    res = {}
    for name, anns in (("real", ann), ("shuffled", shuf), ("mixed", mixed), ("wrong_tool", wrong), ("empty", [])):
        dst, rep = apply_feedback(artifact, anns, name, work, isolation, lab_dirs, python)
        rep["size_after"] = checks.size_check(dst, fp16, budget_fraction); rep["no_new_parts_after"] = checks.no_new_parts(dst, arch, bcfg)
        rep["bytes_added"] = rep["size_after"]["candidate_bytes"] - size0["candidate_bytes"]
        if rep["rc"] != 0: rep["status"] = "feedback_crashed"; out["arms"][name] = rep; continue
        e = ev(dst); rep["status"] = e["status"]; rep["answer_leak_checks"] = {k: e.get(k) for k in ("leak_view", "leak_prompts", "leak_outputs")}
        rep["teach_acc"] = sum(e["cand_correct"][:nt]) / nt if e["status"] == "completed" else None
        res[name] = (dst, e); out["arms"][name] = rep
    out["empty_byte_identical"] = bool(out["arms"]["empty"]["byte_identical"] and out["arms"]["empty"]["tree_sha256_after"] == art_sha)
    ok_arms = all(out["arms"][n].get("status") == "completed" for n in ("real", "shuffled", "mixed", "wrong_tool", "empty"))
    out["arms_completed"] = ok_arms
    if not ok_arms: out["status"] = "failed"; C.write_json(os.path.join(out_dir, "feedback_result.json"), out); return out
    R, S_, E, M_, W_ = (res[n][1] for n in ("real", "shuffled", "empty", "mixed", "wrong_tool"))
    exam_after = None
    if exam_rows:
        ex = exam_gen.evaluate([dict(r, _canary=canary) for r in exam_rows], res["real"][0], base_id, base_texts=exam_base_texts, workdir=work, isolation=isolation,
                               hide_dirs=lab_dirs, python=python, canary=canary)
        if ex["status"] != "completed": raise SystemExit("exam-after failed")
        exam_after = ex["cand_correct"]; out["exam_after_agree_with_before"] = None
    fbblock = dict(n_annotations=len(ann), bytes_added=out["arms"]["real"]["bytes_added"],
                   teach=dict(n=nt, base_correct=before["base_correct"][:nt], real_correct=R["cand_correct"][:nt], shuffled_correct=S_["cand_correct"][:nt], empty_correct=E["cand_correct"][:nt],
                              mixed_label_correct=M_["cand_correct"][:nt], random_wrong_tool_correct=W_["cand_correct"][:nt], cluster=[family(r) for r in teach]),
                   neighbor=dict(n=len(nb), before_correct=before["cand_correct"][nt:], after_correct=R["cand_correct"][nt:]))
    if exam_after is not None: fbblock["exam_after_correct"] = exam_after
    out.update(status="completed", feedback_block=fbblock, teach_before_cand_acc=sum(before["cand_correct"][:nt]) / nt, sizes=dict(before=size0["candidate_bytes"], after=out["arms"]["real"]["size_after"]["candidate_bytes"]),
               neighbors_base_correct_frac=sum(before["base_correct"][nt:]) / len(nb), texts=dict(real=R["cand_texts"][:nt], shuffled=S_["cand_texts"][:nt], empty=E["cand_texts"][:nt], mixed=M_["cand_texts"][:nt], wrong_tool=W_["cand_texts"][:nt], base=bt[:nt]))
    C.write_json(os.path.join(out_dir, "feedback_result.json"), out)
    shutil.rmtree(work, ignore_errors=True); return out


def main():
    ap = argparse.ArgumentParser()
    for k in ("artifact", "base-id", "base-arch", "annotations", "teach", "neighbor", "out-dir"): ap.add_argument("--" + k, required=True)
    ap.add_argument("--exam"); ap.add_argument("--exam-base-texts"); ap.add_argument("--budget-fraction", type=float, default=0.10); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--isolation", default="auto"); ap.add_argument("--allow-locked", action="store_true")
    a = ap.parse_args()
    ex = ebt = None
    if a.exam:
        C.locked_guard(a.exam, a.allow_locked); ex = C.load_jsonl(a.exam); ebt = [o for o in json.load(open(a.exam_base_texts))["base_texts"]]
    o = run_protocol(a.artifact, a.base_id, a.base_arch, dict(ann=a.annotations, teach=a.teach, nb=a.neighbor), a.out_dir, a.budget_fraction, a.seed, a.isolation, exam_rows=ex, exam_base_texts=ebt)
    print("FEEDBACK_RESULT", o.get("status"), "empty_byte_identical", o.get("empty_byte_identical"))


if __name__ == "__main__": main()
