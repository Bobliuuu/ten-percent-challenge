import json, os, subprocess, sys
import pytest
import common3 as C, checks, runner, sandbox, exam_gen, feedback_protocol as FP, final_proof, synth
from conftest import ANSWER

NETNS = sandbox.netns_usable()


def exam_rows(): return synth.make_exam()


def test_sandbox_guard_blocks_network_and_logs():
    r = sandbox.run([sys.executable, "-c", "import socket; socket.create_connection(('1.1.1.1', 443), timeout=2)"], mode="guard")
    assert r["rc"] != 0 and "netguard" in r["stderr"] and r["evidence"]["network"].startswith("blocked") and r["evidence"]["network_attempts_logged"]


@pytest.mark.skipif(not NETNS, reason="needs passwordless sudo + unshare (Linux box)")
def test_sandbox_netns_blocks_network_and_hides_label_dir(tmp_path):
    lab = tmp_path / "labels"; lab.mkdir(); (lab / "gold.jsonl").write_text("secret")
    code = f"import os; print('SEEN', os.path.exists({str(lab / 'gold.jsonl')!r}))"
    r = sandbox.run([sys.executable, "-c", code], mode="netns", hide_dirs=[str(lab)])
    assert "SEEN False" in r["stdout"] and r["evidence"]["network"].startswith("blocked") and r["evidence"]["mode"] == "netns"
    assert (lab / "gold.jsonl").exists()                                   # untouched outside the namespace


def ev(art, base_cmd, rows=None, **k):
    rows = rows or exam_rows()
    return exam_gen.evaluate(rows, art, "stub", base_cmd=base_cmd, isolation="guard", canary="CANARY-t", **k)


def test_evaluate_arrays_and_agree(make_art, base_cmd):
    r = ev(make_art(), base_cmd); n = len(exam_rows())
    assert r["status"] == "completed" and len(r["base_correct"]) == len(r["cand_correct"]) == len(r["agree"]) == n
    assert r["base_correct"] == r["cand_correct"]                            # untaught candidate behaves like the base stub
    assert sum(r["base_correct"]) < n and all(a == 1 for a in r["agree"])    # base stub is wrong on `convert` (arg name), so not 100%
    assert r["leak_view"]["ok"] and r["leak_prompts"]["ok"] and r["leak_outputs"]["ok"] and r["artifact_unchanged_by_answer"]


def test_candidate_never_sees_labels(make_art, base_cmd, tmp_path):
    spy = ANSWER.replace("rows = json.load(open(a.i))", "rows = json.load(open(a.i)); open(os.path.join(os.environ.get('SPY_DIR', '/nonexistent'), 'view_seen.json'), 'w').write(json.dumps(rows))")
    os.environ["SPY_DIR"] = str(tmp_path); ev(make_art(answer=spy), base_cmd); os.environ.pop("SPY_DIR")
    seen = json.load(open(tmp_path / "view_seen.json")); assert all(set(x) == {"id", "query", "tools"} for x in seen)
    assert "gold" not in json.dumps(seen) and "CANARY" not in json.dumps(seen)


def test_cheat_candidate_network_attempt_flagged(make_art, base_cmd):
    cheat = "import socket\ntry:\n    socket.create_connection(('1.1.1.1', 443), timeout=1)\nexcept Exception:\n    pass\n" + ANSWER
    r = ev(make_art(answer=cheat), base_cmd); assert r["status"] == "completed" and r["network_attempts"]          # runtime evidence
    assert not checks.no_new_parts(make_art("art2", answer=cheat), __import__("conftest").BASE_ARCH)["ok"]          # and static grep


def test_candidate_that_edits_artifact_during_answer(make_art, base_cmd):
    bad = ANSWER + "\nopen(os.path.join(a.artifact, 'model.safetensors'), 'ab').write(b'1')\n"
    assert ev(make_art(answer=bad), base_cmd)["artifact_unchanged_by_answer"] is False


def test_candidate_crash_and_bad_output(make_art, base_cmd):
    assert ev(make_art("c1", answer="raise SystemExit(3)\n"), base_cmd)["status"] == "candidate_crashed"
    bad = ANSWER.replace('out.append({"id": r["id"],', 'out.append({"id": "WRONG",')
    assert ev(make_art("c2", answer=bad), base_cmd)["status"] == "candidate_bad_output"


def test_view_leak_blocks_run(make_art, base_cmd):
    rows = exam_rows(); rows[0]["query"] += ' {"name": "get_weather", "arguments": {"city": "Paris"}}'; rows[0]["gold"] = [dict(name="get_weather", arguments=dict(city="Paris"))]
    assert ev(make_art(), base_cmd, rows=rows)["status"] == "leak_in_view"


def test_prompt_leak_detected(make_art, base_cmd):
    rows = exam_rows(); g = json.dumps(dict(name=rows[9]["gold"][0]["name"], arguments=rows[9]["gold"][0]["arguments"]))
    dump = ANSWER.replace("json.dump([r[\"query\"] for r in rows]", f"json.dump([r[\"query\"] + {g!r} for r in rows]")
    assert dump != ANSWER and not ev(make_art(answer=dump), base_cmd)["leak_prompts"]["ok"]


def test_single_use_claim(tmp_path):
    exam_gen.claim(str(tmp_path), "t", 1, {}); 
    with pytest.raises(SystemExit): exam_gen.claim(str(tmp_path), "t", 1, {})
    exam_gen.claim(str(tmp_path), "t", 2, {})                                  # other stage ok
    exam_gen.claim(str(tmp_path), "u", 1, {})                                  # other tag ok


def test_claim_release_after_infra_failure(tmp_path):
    p = exam_gen.claim(str(tmp_path), "t", 1, {}); exam_gen.release(p); exam_gen.claim(str(tmp_path), "t", 1, {})


def test_exam_cli_refuses_locked(tmp_path):
    p = tmp_path / "locked.jsonl"; p.write_text("x")
    r = subprocess.run([sys.executable, exam_gen.__file__, "--exam", str(p), "--artifact", str(tmp_path), "--base-id", "x", "--base-arch", "x", "--tag", "t", "--out-dir", str(tmp_path / "o")], capture_output=True, text=True)
    assert r.returncode != 0 and "LOCKED" in r.stderr and not (tmp_path / "o").exists()


# ---------------------------------------------------------------- feedback protocol
def test_validate_sets_negatives(pools):
    _, (ann, teach, nb) = pools
    assert FP.validate_sets(ann, teach, nb) == []
    assert any("teach-set must be" in p for p in FP.validate_sets(ann, teach[:10], nb))
    assert any("annotations must be" in p for p in FP.validate_sets(ann * 5, teach, nb))
    assert any("share ids" in p for p in FP.validate_sets(ann, teach + [dict(ann[0])], nb))
    dup = dict(teach[0], id="new"); assert any("content" in p for p in FP.validate_sets(ann + [dup], teach, nb))
    assert any("share" in p for p in FP.validate_sets(ann, teach, nb, exam_rows=[dict(teach[0], id="x")]))      # exam overlap
    assert any("without gold" in p for p in FP.validate_sets(ann, [dict(teach[0], gold=[])] + teach[1:], nb))


def test_shuffle_is_derangement_and_deterministic(pools):
    _, (ann, _, _) = pools; s, why = FP.shuffled_annotations(ann, 0); s2, _ = FP.shuffled_annotations(ann, 0)
    assert why is None and s == s2 and all(FP.gold_key(a) != FP.gold_key(b) for a, b in zip(ann, s))
    assert sorted(map(FP.gold_key, s)) == sorted(map(FP.gold_key, ann))        # same multiset of labels, same prompts
    assert [a["query"] for a in s] == [a["query"] for a in ann]
    assert FP.shuffled_annotations([ann[0], dict(ann[0], id="z")], 0)[0] is None


def test_lock_written_before_feedback_and_immutable(pools, make_art, tmp_path):
    files, (ann, teach, nb) = pools; art = make_art(); p = str(tmp_path / "LOCK.json")
    FP.write_lock(p, files, ann, teach, nb, C.tree_sha256(art), {}); assert not os.access(p, os.W_OK)
    FP.write_lock(p, files, ann, teach, nb, C.tree_sha256(art), {})            # same inputs: fine
    C.dump_jsonl(teach[:-1], files["teach"])                                    # teach-set quietly changed after the lock
    with pytest.raises(SystemExit): FP.write_lock(p, files, ann, teach[:-1], nb, C.tree_sha256(art), {})


def test_apply_feedback_empty_is_byte_identical_and_real_changes(pools, make_art, tmp_path):
    _, (ann, _, _) = pools; art = make_art(); w = str(tmp_path / "w"); os.makedirs(w); before = C.tree_sha256(art)
    d0, r0 = FP.apply_feedback(art, [], "empty", w, "guard", [], sys.executable); d1, r1 = FP.apply_feedback(art, ann, "real", w, "guard", [], sys.executable)
    assert r0["byte_identical"] and r0["tree_sha256_after"] == before and r1["changed_files"] == ["model.safetensors"] and not r1["byte_identical"]
    assert C.tree_sha256(art) == before                                        # original never touched


def test_empty_feedback_that_secretly_edits_is_caught(pools, make_art, tmp_path):
    bad = "import argparse, os\nap = argparse.ArgumentParser(); ap.add_argument('--artifact'); ap.add_argument('--annotations'); a = ap.parse_args()\nopen(os.path.join(a.artifact, 'model.safetensors'), 'ab').write(b'0')\n"
    art = make_art(feedback=bad); w = str(tmp_path / "w"); os.makedirs(w)
    _, r = FP.apply_feedback(art, [], "empty", w, "guard", [], sys.executable); assert r["byte_identical"] is False


def test_full_final_proof_plumbing(pools, make_art, base_cmd, exam, tmp_path, arch_file):
    files, _ = pools; art = make_art(); out = str(tmp_path / "run")
    r = subprocess.run([sys.executable, os.path.join(C.HERE, "final_proof.py"), "--exam", exam, "--artifact", art, "--base-id", "stub", "--base-arch", arch_file, "--annotations", files["ann"],
                        "--teach", files["teach"], "--neighbor", files["nb"], "--tag", "t", "--out-dir", out, "--isolation", "guard", "--base-cmd-json", json.dumps(base_cmd)], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-1500:] + r.stderr[-2500:]
    s = json.load(open(f"{out}/t.PROOF_SUMMARY.json")); inp = json.load(open(f"{out}/t.scorecard_input.json")); sc = json.load(open(f"{out}/t.scorecard.json"))
    assert s["teach_acc"] == dict(base=0.0, real=1.0, shuffled=0.0, empty=0.0) and s["empty_byte_identical"] is True
    assert s["REAL_RESULT"] is False                                           # not a locked run, guard isolation
    fb = inp["feedback"]; assert fb["teach"]["n"] == 30 and len(fb["neighbor"]["after_correct"]) == 30 and len(fb["exam_after_correct"]) == len(inp["base_correct"])
    assert fb["neighbor"]["before_correct"] == fb["neighbor"]["after_correct"] and fb["bytes_added"] == 0
    assert set(inp["source"]) == {"S1", "S2"} and set(inp["kind"]) == {"nocall", "single", "multi"}
    assert "teach_clear_gain(lo>0,mean>=.15)" not in sc["failed_gates"] and "controls_clearly_worse(lo>0)" not in sc["failed_gates"]
    assert "exam>=2000" in sc["failed_gates"]                                   # tiny synthetic exam must fail the size gate of the exam, as it should
    assert "size<=10%" not in sc["failed_gates"] and "no_new_parts" not in sc["failed_gates"]
    # single use: re-running the same tag is refused
    r2 = subprocess.run(r.args, capture_output=True, text=True); assert r2.returncode != 0 and "single-use" in (r2.stderr + r2.stdout)
    assert s["feedback_arms"]["empty"]["changed_files"] == [] and s["feedback_arms"]["real"]["changed_files"] == ["model.safetensors"]
