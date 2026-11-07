"""CPU unit tests on a tiny random Qwen3-architecture model + the real Qwen3 tokenizer.  Env TPQ_TOKDIR = dir with tokenizer.json, tokenizer_config.json, config.json, generation_config.json
(default: scratchpad copy; any Qwen3 snapshot dir works).   pytest -q v2/artifact/tests"""
import hashlib, json, math, os, shutil, subprocess, sys
import numpy as np, pytest, torch
import loader as LD, quantize as QZ, writer as WR, make_tiny, checks

TOK = os.environ.get("TPQ_TOKDIR", "/private/tmp/claude-501/-Users-jhome/76509dd1-0c3e-4650-80c1-859ea89403f6/scratchpad/tiny/qwen3_8b_meta")
PY = sys.executable


def tree_sha(d):
    h = hashlib.sha256()
    for r, _, fs in sorted(os.walk(d)):
        for f in sorted(fs): h.update(os.path.relpath(os.path.join(r, f), d).encode() + hashlib.sha256(open(os.path.join(r, f), "rb").read()).digest())
    return h.hexdigest()


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    d = tmp_path_factory.mktemp("tpq"); base = str(d / "base"); art = str(d / "art"); sim = str(d / "sim")
    make_tiny.make(base, TOK, layers=2)
    LD.TARGET_SYM = 1 << 16                                                  # force many chunks per tensor
    rep = QZ.quantize_model(base, art, dict(attn=3, o=3, mlp=3, down=3, emb=4, head=4), g=256, g_emb=128, mbits=8, sim_dir=sim, dev="cpu", log=lambda *a: None)
    WR.assemble(art)
    arch = checks.arch_of_dir(base); fp16 = sum(2 * math.prod(s) for _, s in arch.values())
    items = [dict(id=f"i{i}", query=f"what is the weather in city number {i}?", tools=[dict(name="get_weather", description="weather", parameters=dict(type="object", properties=dict(city=dict(type="string")), required=["city"]))]) for i in range(5)]
    json.dump(items, open(d / "view.json", "w"))
    return dict(d=d, base=base, art=art, sim=sim, arch=arch, fp16=fp16, rep=rep, items=items)


def test_bitexact_vs_sim(env):
    import verify
    r = verify.bitexact(env["art"], env["sim"], "cpu", 4)
    assert r["all_equal"] and r["tensors_compared"] == 2 * 7 + 2


def test_codec_roundtrip_random(tmp_path):
    rng = np.random.RandomState(0); codes = rng.randint(0, 5, (50, 512)).astype(np.uint8); sc = (rng.rand(50, 4).astype(np.float32) + 0.1)
    sc = LD.round_scale_mbits(torch.tensor(sc), 6).numpy(); w = LD.Writer(str(tmp_path / "x.tpq")); w.add_signs("k", torch.ones(512)); w.add_q("t", codes, sc, 5, 128, 6, "k", tgt=1024); w.finish()
    c = LD.Container(str(tmp_path / "x.tpq")); cd, s2 = c.read_q("t"); assert (cd.numpy() == codes).all() and (s2.numpy() == sc.astype(np.float16)).all() and len(c.T["t"]["lens"]) // 3 == 25


def test_scale_with_unrounded_mantissa_is_refused():
    with pytest.raises(AssertionError): LD.encode_chunk(np.zeros((1, 128), np.uint8), np.array([[0.1234567]], np.float16), [.5, .5], [1 / 32] * 32, 6)


def test_harness_checks(env):
    sz = checks.size_check(env["art"], env["fp16"]); nnp = checks.no_new_parts(env["art"], env["arch"], json.load(open(env["base"] + "/config.json")))
    assert nnp["ok"], nnp; assert nnp["tensor_source"] == "declared_manifest" and not nnp["extra_files"] and nnp["grep_train_hits"] == 0 and nnp["network_hits"] == 0


def test_manifest_equals_container(env):
    c = LD.Container(os.path.join(env["art"], LD.BLOB)); man = json.load(open(os.path.join(env["art"], "tensor_manifest.json")))
    assert set(man) == set(env["arch"]) and all(man[k] == [env["arch"][k][0], env["arch"][k][1]] for k in man) and all(c.T[k]["shape"] == man[k][1] for k in man)


def run_answer(art, view, out, extra=()):
    return subprocess.run([PY, os.path.join(art, "answer.py"), "--artifact", art, "--in", view, "--out", out, "--max-new-tokens", "10", "--bs", "2", "--device", "cpu", *extra], capture_output=True, text=True)


def test_answer_equals_sim_path_and_leaves_tree_untouched(env):
    from safetensors.torch import load_file
    from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
    before = tree_sha(env["art"]); out = str(env["d"] / "o.json"); r = run_answer(env["art"], str(env["d"] / "view.json"), out, ["--dump-prompts", str(env["d"] / "p.json")]); assert r.returncode == 0, r.stderr[-2000:]
    assert tree_sha(env["art"]) == before                                     # answer does not modify the artifact (no pyc, no cache)
    real = json.load(open(out)); assert [x["id"] for x in real] == [i["id"] for i in env["items"]]
    sd = {}
    for f in sorted(os.listdir(env["sim"])): sd.update(load_file(os.path.join(env["sim"], f)))
    for k, v in load_file(os.path.join(env["base"], "model.safetensors")).items(): sd.setdefault(k, v)       # raw (unquantised) tensors come from the base
    conf = AutoConfig.from_pretrained(env["base"]); m = AutoModelForCausalLM.from_config(conf, dtype=torch.bfloat16); m.load_state_dict(sd); m.eval()
    tok = AutoTokenizer.from_pretrained(env["base"]); tok.padding_side = "left"
    txt, _ = LD.generate_with(m, tok, env["items"], 10, 2, dev="cpu")
    assert txt == [x["text"] for x in real]


def test_empty_feedback_is_byte_identical(env, tmp_path):
    art = str(tmp_path / "a"); shutil.copytree(env["art"], art); open(tmp_path / "e.jsonl", "w").write(""); before = tree_sha(art)
    r = subprocess.run([PY, os.path.join(art, "feedback.py"), "--artifact", art, "--annotations", str(tmp_path / "e.jsonl")], capture_output=True, text=True); assert r.returncode == 0, r.stderr[-1500:]
    assert "FEEDBACK_NOOP" in r.stdout and tree_sha(art) == before


def test_feedback_edits_in_place_same_files_small_size_change(env, tmp_path):
    art = str(tmp_path / "a"); shutil.copytree(env["art"], art)
    ann = [dict(id="a1", query="weather in paris please", tools=env["items"][0]["tools"], gold=[dict(name="get_weather", arguments=dict(city="Paris"))])]
    open(tmp_path / "ann.jsonl", "w").write("\n".join(json.dumps(a) for a in ann)); files0 = sorted(os.listdir(art)); size0 = sum(os.path.getsize(os.path.join(art, f)) for f in files0); sha0 = {f: hashlib.sha256(open(os.path.join(art, f), "rb").read()).hexdigest() for f in files0}
    r = subprocess.run([PY, os.path.join(art, "feedback.py"), "--artifact", art, "--annotations", str(tmp_path / "ann.jsonl"), "--device", "cpu"], capture_output=True, text=True); assert r.returncode == 0, r.stderr[-2000:]
    assert "FEEDBACK_DONE" in r.stdout, r.stdout
    assert sorted(os.listdir(art)) == files0                                  # zero added files (no .tmp left)
    changed = [f for f in files0 if hashlib.sha256(open(os.path.join(art, f), "rb").read()).hexdigest() != sha0[f]]; assert changed == ["model.tpq"]
    size1 = sum(os.path.getsize(os.path.join(art, f)) for f in files0); assert abs(size1 - size0) / size0 < 1e-4, (size0, size1)
    nnp = checks.no_new_parts(art, env["arch"], json.load(open(env["base"] + "/config.json"))); assert nnp["ok"]
    # only lm_head rows changed: every other tensor decodes identically
    c0 = LD.Container(os.path.join(env["art"], LD.BLOB)); c1 = LD.Container(os.path.join(art, LD.BLOB)); diff = []
    for n in c0.names():
        if c0.T[n]["kind"] == "raw": assert torch.equal(c0.read_raw(n), c1.read_raw(n))
        else:
            a0, s0 = c0.read_q(n); a1, s1 = c1.read_q(n)
            if not (torch.equal(a0, a1) and torch.equal(s0, s1)): diff.append((n, int((a0 != a1).any(1).sum())))
    assert [d[0] for d in diff] == ["lm_head.weight"] and diff[0][1] > 0
    r2 = run_answer(art, str(env["d"] / "view.json"), str(tmp_path / "o2.json")); assert r2.returncode == 0, r2.stderr[-1500:]


def test_forbidden_words_absent_from_shipped_code(env):
    import re
    for f in ("answer.py", "feedback.py"):
        t = open(os.path.join(env["art"], f)).read()
        for rg in (checks.TRAIN_RE, checks.NET_RE, checks.CONSTRAINED_RE): assert not rg.search(t), (f, rg.pattern, rg.search(t))


def test_tied_checkpoint_with_duplicate_lm_head_uses_alias(tmp_path):
    base = str(tmp_path / "b"); art = str(tmp_path / "a"); sim = str(tmp_path / "s"); make_tiny.make(base, TOK, layers=1, tied=True)
    QZ.quantize_model(base, art, dict(attn=3, o=3, mlp=3, down=3, emb=4, head=4), sim_dir=sim, dev="cpu", log=lambda *a: None); WR.assemble(art)
    c = LD.Container(os.path.join(art, LD.BLOB)); assert c.T["lm_head.weight"]["kind"] == "alias" and c.T["model.embed_tokens.weight"]["kind"] == "q"
    import verify; assert verify.bitexact(art, sim, "cpu", 2)["all_equal"]
    arch = checks.arch_of_dir(base); assert checks.no_new_parts(art, arch, json.load(open(base + "/config.json")))["ok"]
    m, _ = LD.load_model(art, "cpu", 2, log=lambda *a: None); assert torch.equal(m.lm_head.weight, m.model.embed_tokens.weight)
