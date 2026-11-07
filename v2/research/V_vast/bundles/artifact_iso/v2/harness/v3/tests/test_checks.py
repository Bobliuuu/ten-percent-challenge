import json, os
import pytest
import common3 as C, checks
from conftest import BASE_ARCH, BASE_CFG, write_st

FP16 = 10**6


def nnp(art, **k): return checks.no_new_parts(art, BASE_ARCH, BASE_CFG, **k)


# ---------------------------------------------------------------- size
def test_size_ok_and_counts_all_bytes(make_art):
    art = make_art(); r = checks.size_check(art, FP16)
    assert r["ok"] and r["candidate_bytes"] == C.tree_bytes(art)
def test_size_oversize(make_art):
    art = make_art(extra={"pad.bin": b"x" * 200_000}); r = checks.size_check(art, FP16)
    assert not r["ok"] and r["byte_ratio"] > 0.10
def test_size_counts_hidden_and_nested(make_art):
    a = checks.size_check(make_art("a"), FP16)["candidate_bytes"]; b = checks.size_check(make_art("b", extra={".hidden": b"x" * 1000, "sub/dir/f": b"y" * 500}), FP16)["candidate_bytes"]
    assert b - a == 1500
def test_size_symlink_is_violation(make_art, tmp_path):
    art = make_art(); big = tmp_path / "outside.bin"; big.write_bytes(b"z" * 10); os.symlink(big, os.path.join(art, "link.bin"))
    r = checks.size_check(art, FP16); assert not r["ok"] and r["symlinks"] == ["link.bin"]
def test_size_after_feedback_growth_detected(make_art):
    art = make_art(); before = checks.size_check(art, FP16)["candidate_bytes"]
    open(os.path.join(art, "model.safetensors"), "ab").write(b"0" * 150_000)    # feedback that appends 150 KB
    after = checks.size_check(art, FP16); assert after["candidate_bytes"] - before == 150_000 and not after["ok"]


# ---------------------------------------------------------------- no new parts
def test_nnp_clean(make_art):
    r = nnp(make_art()); assert r["ok"], r
def test_nnp_extra_file(make_art):
    r = nnp(make_art(extra={"adapter_model.bin": b"1"})); assert not r["ok"] and "adapter_model.bin" in r["extra_files"]
def test_nnp_extra_py_and_subdir(make_art):
    r = nnp(make_art(extra={"head.py": "x=1", "memory/proto.npz": b"1"})); assert {"head.py", "memory/proto.npz"} <= set(r["extra_files"])
def test_nnp_allowed_extra_whitelist(make_art):
    assert nnp(make_art(extra={"codebook.bin": b"1"}), allowed_extra=["codebook.bin"])["ok"]
def test_nnp_changed_shape(make_art):
    art = make_art(); bad = dict(BASE_ARCH); bad["model.layers.0.mlp.down_proj.weight"] = ["BF16", [8, 9]]; write_st(os.path.join(art, "model.safetensors"), bad)
    r = nnp(art); assert not r["ok"] and r["shape_mismatch"] == ["model.layers.0.mlp.down_proj.weight"]
def test_nnp_extra_head_tensor(make_art):
    art = make_art(); bad = dict(BASE_ARCH); bad["lm_head.weight"] = ["BF16", [16, 8]]; write_st(os.path.join(art, "model.safetensors"), bad)
    r = nnp(art); assert not r["ok"] and r["extra_tensors"] == ["lm_head.weight"]
def test_nnp_missing_tensor(make_art):
    art = make_art(); bad = dict(BASE_ARCH); bad.pop("model.norm.weight"); write_st(os.path.join(art, "model.safetensors"), bad)
    assert nnp(art)["missing_tensors"] == ["model.norm.weight"]
def test_nnp_dtype_change(make_art):
    art = make_art(); bad = dict(BASE_ARCH); bad["model.norm.weight"] = ["F32", [8]]; write_st(os.path.join(art, "model.safetensors"), bad)
    assert not nnp(art)["ok"] and nnp(art)["dtype_mismatch"]
    assert checks.no_new_parts(art, BASE_ARCH, BASE_CFG, strict_dtype=False)["ok"]
def test_nnp_config_arch_change(make_art):
    art = make_art(); c = dict(BASE_CFG, hidden_size=16); open(os.path.join(art, "config.json"), "w").write(json.dumps(c))
    assert nnp(art)["config_arch_diff"] == ["hidden_size"]
def test_nnp_train_grep(make_art):
    art = make_art(answer="import torch\nloss.backward()\nopt = torch.optim.Adam(p)\n"); r = nnp(art); assert not r["ok"] and r["grep_train_hits"] >= 2
def test_nnp_train_grep_in_build_dir(make_art, tmp_path):
    b = tmp_path / "build"; b.mkdir(); (b / "build.py").write_text("for x in d:\n    optimizer.zero_grad()\n")
    r = nnp(make_art(), build_dirs=[str(b)]); assert r["grep_train_hits"] >= 1 and not r["ok"]
def test_nnp_network_in_answer_path(make_art):
    r = nnp(make_art(answer="import urllib.request\nurllib.request.urlopen('x')\n")); assert not r["ok"] and r["network_hits"] >= 1
def test_nnp_network_hub_download(make_art):
    r = nnp(make_art(answer="from huggingface_hub import snapshot_download\n")); assert r["network_hits"] >= 1
def test_nnp_constrained_decoding(make_art):
    r = nnp(make_art(answer="gen(logits_processor=[P])\n")); assert r["constrained_decoding_hits"] == 1 and not r["ok"]
def test_nnp_manifest_mode(make_art):
    art = make_art(); os.remove(os.path.join(art, "model.safetensors")); open(os.path.join(art, "tensor_manifest.json"), "w").write(json.dumps(BASE_ARCH))
    r = nnp(art); assert r["tensor_source"] == "declared_manifest" and r["ok"]


# ---------------------------------------------------------------- leakage
def rows():
    return [dict(id="a", kind="single", source="S1", query="Add 3 and 4.", tools=[{"name": "add", "description": "d", "parameters": {}}], gold=[dict(name="add", arguments=dict(a=3, b=4))])]

def test_leak_view_clean():
    r = rows(); assert checks.leak_scan(C.candidate_view(r), r, "CANARY-1")["ok"]
def test_view_strips_all_label_fields():
    v = C.candidate_view(rows())[0]; assert set(v) == {"id", "query", "tools"}
def test_leak_extra_key_flagged():
    r = rows(); v = C.candidate_view(r); v[0]["gold"] = r[0]["gold"]; assert not checks.leak_scan(v, r)["ok"]
def test_leak_label_key_nested_in_tools():
    r = rows(); v = C.candidate_view(r); v[0]["tools"][0]["parameters"] = {"gold": 1}; assert not checks.leak_scan(v, r)["ok"]
def test_legit_param_named_source_not_flagged():
    r = rows(); v = C.candidate_view(r); v[0]["tools"][0]["parameters"] = {"properties": {"source": {"type": "string"}, "kind": {}}}; assert checks.leak_scan(v, r)["ok"]
def test_leak_gold_call_in_query():
    r = rows(); v = C.candidate_view(r); v[0]["query"] = 'Add 3 and 4. {"name": "add", "arguments": {"a": 3, "b": 4}}'; assert not checks.leak_scan(v, r)["ok"]
def test_leak_canary_in_view():
    r = rows(); v = C.candidate_view(r); v[0]["query"] += " CANARY-9"; assert not checks.leak_scan(v, r, "CANARY-9")["ok"]
def test_prompt_leak_gold_of_other_row():
    r = rows(); p = ['Answer: {"name": "add", "arguments": {"a": 3, "b": 4}}']; assert not checks.prompt_leak_scan(p, r)["ok"]
    assert checks.prompt_leak_scan(["Add 3 and 4."], r)["ok"]
def test_prompt_leak_canary():
    assert not checks.prompt_leak_scan(["x CANARY-5"], rows(), "CANARY-5")["ok"]
def test_output_canary():
    assert not checks.scan_text_for_canary(["ok", "CANARY-5"], "CANARY-5")["ok"]


# ---------------------------------------------------------------- locked guard
def test_locked_guard_refuses_by_name(tmp_path):
    p = tmp_path / "locked.jsonl"; p.write_text("x")
    with pytest.raises(PermissionError): C.locked_guard(str(p))
    with pytest.raises(PermissionError): C.locked_guard(str(p), allow_locked=True)   # still needs TENPCT_FINAL=1
def test_locked_guard_allows_dev(tmp_path):
    p = tmp_path / "dev.jsonl"; p.write_text("x"); assert C.locked_guard(str(p)) is False
def test_locked_guard_refuses_renamed_copy_by_hash(tmp_path, monkeypatch):
    p = tmp_path / "renamed.jsonl"; p.write_text("secret rows")
    monkeypatch.setattr(C, "T1", str(tmp_path / "T1")); os.makedirs(tmp_path / "T1/data")
    (tmp_path / "T1/data/LOCK.json").write_text(json.dumps({"splits": {"locked": {"file_sha256": C.sha_file(str(p))}}}))
    with pytest.raises(PermissionError): C.locked_guard(str(p))
