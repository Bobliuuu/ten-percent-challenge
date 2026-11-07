"""Low-level generation evaluation used by the exam runner and the feedback protocol.
  run_base(...)       base model in its own process (network allowed, sees only the view)
  run_candidate(...)  artifact `answer` command in the sandbox (network off, label dirs hidden); artifact tree hash must not change
  score(...)          T1 scorer -> per-item arrays (labels live only in this orchestrator process)"""
import json, os, subprocess, sys, time
import common3 as C, sandbox


def _check_out(out, view):
    ids = [r["id"] for r in view]
    if not isinstance(out, list) or [o.get("id") for o in out] != ids: raise ValueError("answer output ids/order differ from the input")
    return [str(o.get("text", "")) for o in out]


def run_base(base_id, view, workdir, python=sys.executable, base_cmd=None, max_new_tokens=512, bs=8):
    """base_cmd (tests): argv template with {in} {out} {base}; default = base_gen.py."""
    inp, outp = os.path.join(workdir, "base_in.json"), os.path.join(workdir, "base_out.json"); json.dump(view, open(inp, "w"))
    argv = [a.format(**{"in": inp, "out": outp, "base": base_id}) for a in base_cmd] if base_cmd else \
           [python, os.path.join(C.HERE, "base_gen.py"), "--base", base_id, "--in", inp, "--out", outp, "--max-new-tokens", str(max_new_tokens), "--bs", str(bs)]
    t0 = time.time(); r = subprocess.run(argv, capture_output=True, text=True)
    if r.returncode != 0: raise RuntimeError("base run failed: " + r.stderr[-1500:])
    return dict(texts=_check_out(json.load(open(outp)), view), seconds=time.time() - t0, out_sha256=C.sha_file(outp))


def run_candidate(artifact, view, workdir, mode="auto", hide_dirs=(), python=sys.executable, dump_prompts=True, extra_args=()):
    """Runs ART/answer.py inside the sandbox. Returns dict(status, texts, evidence, stderr_tail, prompts, artifact_unchanged, seconds)."""
    inp, outp, pr = os.path.join(workdir, "cand_in.json"), os.path.join(workdir, "cand_out.json"), os.path.join(workdir, "cand_prompts.json")
    json.dump(view, open(inp, "w")); art = os.path.abspath(artifact); before = C.tree_sha256(art)
    argv = [python, os.path.join(art, "answer.py"), "--artifact", art, "--in", inp, "--out", outp, *extra_args]
    if dump_prompts: argv += ["--dump-prompts", pr]
    t0 = time.time(); r = sandbox.run(argv, mode=mode, hide_dirs=hide_dirs, python=python)
    res = dict(status="completed", evidence=r["evidence"], seconds=time.time() - t0, stderr_tail=r["stderr"][-1500:],
               artifact_unchanged=C.tree_sha256(art) == before, texts=None, prompts=None)
    if r["rc"] != 0 or not os.path.exists(outp):
        res.update(status="candidate_crashed", rc=r["rc"]); return res
    try: res["texts"] = _check_out(json.load(open(outp)), view)
    except Exception as e: res.update(status="candidate_bad_output", error=str(e)); return res
    res["out_sha256"] = C.sha_file(outp)
    if os.path.exists(pr): res["prompts"] = json.load(open(pr))
    return res


def score(rows, texts, scorer=None):
    S = scorer or C.get_scorer(); res = [S.score_item(t, r) for t, r in zip(texts, rows)]
    return dict(correct=[int(x["correct"]) for x in res], parse_fail=[int(x["parse_fail"]) for x in res], n_calls=[x["n_calls"] for x in res])


def agree(a, b): return [int(x.strip() == y.strip()) for x, y in zip(a, b)]
