"""Unit tests for scorecard v2 (run: cd harness/unit && python3 -m pytest -q)."""
import os, sys, json, tempfile
import numpy as np, pytest
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import scorecard as SC

def make_tree(tmp, extra=False, leak=None):
    d = os.path.join(tmp, "art"); os.makedirs(d, exist_ok=True)
    open(f"{d}/model.safetensors", "wb").write(os.urandom(5000)); open(f"{d}/config.json", "w").write('{"a":1}')
    if extra: open(f"{d}/head.npz", "wb").write(b"zzzz")
    if leak: open(f"{d}/model.safetensors", "ab").write(leak.encode())
    return d

def synth(tmp, ret_cells=None, n_s1=1300, base_acc=0.8, seed=0, extra=False, leak=None, runs=True, feedback=None):
    g = np.random.default_rng(seed); src, kind, b, c = [], [], [], []
    cellsz = {("S1", "nocall"): 400, ("S1", "single"): 450, ("S1", "multi"): 450, ("S2", "nocall"): 400, ("S2", "single"): 500, ("S2", "multi"): 372}
    for (s, k), m in cellsz.items():
        for _ in range(m):
            bi = g.random() < base_acc; ci = bi and (g.random() < (ret_cells or {}).get((s, k), 1.0)); src.append(s); kind.append(k); b.append(bool(bi)); c.append(bool(ci))
    d = make_tree(tmp, extra, leak)
    R = dict(cand_correct=c, kind=kind, source=src, size=dict(cand_bytes=9.5, base_fp16_bytes=100), chance=0.1, generative=True, kinds_real_output=True,
             artifact_tree=dict(dir=d, allowed_manifest={"model.safetensors": None, "config.json": None}, needles=["what is the capital of the moon base alpha"]))
    if runs:
        # K=3 base runs that differ slightly (engine/batch); base_correct = intersection
        R["base_runs"] = [dict(correct=b, engine="vllm", batch=32), dict(correct=b, engine="hf", batch=8), dict(correct=b, engine="vllm", batch=1)]
    else: R["base_correct"] = b
    if feedback: R["feedback"] = feedback
    return R

def fb_block(n=60, clusters=12, draws=3, real=0.8, base=0.3, ctl=0.3, seed=1):
    g = np.random.default_rng(seed); N = n * draws; cl = np.tile(np.arange(n) % clusters, draws); dr = np.repeat(np.arange(draws), n)
    mk = lambda p: (g.random(N) < p).astype(int).tolist()
    return dict(n_annotations=12, bytes_added=0, teach=dict(n=n, base_correct=mk(base), real_correct=mk(real), empty_correct=mk(ctl), shuffled_correct=mk(ctl), mixed_label_correct=mk(ctl), random_wrong_tool_correct=mk(ctl), cluster=cl.tolist(), draw=dr.tolist()),
                neighbor=dict(n=400, before_correct=[1] * 400, after_correct=[1] * 400), exam_after_correct=None)

@pytest.fixture()
def tmp(): 
    with tempfile.TemporaryDirectory() as d: yield d

def test_perfect_candidate_passes_exam_gates(tmp):
    o = SC.score(synth(tmp)); assert [k for k in o["failed_gates"] if k != "feedback_present"] == [], o["failed_gates"]

def test_source_kind_cell_collapse_fails(tmp):
    o = SC.score(synth(tmp, ret_cells={("S2", "nocall"): 0.88})); assert "every_source_kind_cell_retention>=95%" in o["failed_gates"]
    assert o["details"]["retention_per_cell"]["S2/nocall"]["ret"] < 0.95

def test_ood_source_gate(tmp):
    o = SC.score(synth(tmp, n_s1=1300, ret_cells={("S2", "single"): 0.92, ("S2", "multi"): 0.92, ("S2", "nocall"): 0.92})); assert "every_source_retention>=95%" in o["failed_gates"]

def test_retention_with_and_without_s1_nocall_reported(tmp):
    o = SC.score(synth(tmp, ret_cells={("S1", "single"): 0.95, ("S1", "multi"): 0.95}))
    d = o["details"]; assert "retention_overall" in d and "retention_excl_S1_nocall" in d and d["retention_excl_S1_nocall"]["ret"] < d["retention_overall"]["ret"]

def test_extra_file_from_tree_not_declared(tmp):
    R = synth(tmp, extra=True); R["no_new_parts"] = dict(extra_files=[], grep_train_hits=0)       # lie
    o = SC.score(R); assert o["gates"]["no_new_parts"] is False and "head.npz" in o["details"]["artifact_tree_audit"]["extra_files"]

def test_grep_hit_from_tree(tmp):
    o = SC.score(synth(tmp, leak="xx what is the capital of the moon base alpha xx")); assert o["gates"]["no_new_parts"] is False and o["details"]["artifact_tree_audit"]["grep_train_hits"] == 1

def test_grep_hit_across_chunk_boundary(tmp):
    d = os.path.join(tmp, "a"); os.makedirs(d); n = "UNIQUE-NEEDLE-ABCDEFGH"
    open(f"{d}/w.bin", "wb").write(b"0" * ((1 << 25) - 5) + n.encode() + b"1" * 100)
    a = SC.audit_artifact_tree(d, {"w.bin": None}, needles=[n]); assert a["grep_train_hits"] == 1

def test_missing_tree_fails_closed(tmp):
    R = synth(tmp); del R["artifact_tree"]; R["no_new_parts"] = dict(extra_files=[], grep_train_hits=0)
    assert SC.score(R)["gates"]["no_new_parts"] is False

def test_no_needles_fails_closed(tmp):
    R = synth(tmp); R["artifact_tree"]["needles"] = []; assert SC.score(R)["gates"]["no_new_parts"] is False

def test_pinned_sha_modified(tmp):
    R = synth(tmp); R["artifact_tree"]["allowed_manifest"] = {"model.safetensors": None, "config.json": "0" * 64}
    assert SC.score(R)["gates"]["no_new_parts"] is False

def test_single_base_run_fails_noise_floor_gates(tmp):
    o = SC.score(synth(tmp, runs=False)); assert "noise_floor_runs>=3" in o["failed_gates"]

def test_identical_config_runs_fail_gate(tmp):
    R = synth(tmp); 
    for r in R["base_runs"]: r["engine"], r["batch"] = "vllm", 32
    assert "noise_floor_runs_vary_engine_or_batch" in SC.score(R)["failed_gates"]

def test_noise_floor_candidate_below_floor_fails(tmp):
    R = synth(tmp); b = np.array(R["base_runs"][0]["correct"]); g = np.random.default_rng(3)
    # base runs differ on 1% of items; candidate = base run 0 -> retention vs intersection is 1.0 but floor mean_pair ~0.99
    runs = []
    for i in range(3):
        x = b.copy(); idx = g.choice(len(x), 40, replace=False); x[idx] = ~x[idx]; runs.append(dict(correct=x.tolist(), engine=["vllm", "hf", "vllm"][i], batch=[32, 8, 1][i]))
    R["base_runs"] = runs; o = SC.score(R); fl = o["details"]["noise_floor"]; assert fl["mean_pair"] < 0.985 and fl["flip_items"] > 40

def test_wilson_power_note(tmp):
    o = SC.score(synth(tmp)); p = o["details"]["power_note"]
    assert p["overall"]["needed_retention"] > 0.98 and p["cell:S2/multi"]["needed_retention"] > 0.95
    assert SC.needed_for_wilson(100, 0.98) is None   # n=100 can never certify >=98% (one break -> lo 0.946)
    k, r = SC.needed_for_wilson(300, 0.95); assert SC.wilson_lo(k, 300) >= 0.95 > SC.wilson_lo(k - 1, 300)

def test_controls_require_10pt_and_lo_positive(tmp):
    R = synth(tmp, feedback=fb_block(ctl=0.3)); R["feedback"]["exam_after_correct"] = R["base_runs"][0]["correct"]
    o = SC.score(R); assert o["gates"]["controls_clearly_worse(>=10pt,lo>0)"] and o["gates"]["controls_complete(empty,shuffled|mixed,random_wrong_tool)"]
    R2 = synth(tmp, feedback=fb_block(real=0.8, ctl=0.72)); R2["feedback"]["exam_after_correct"] = R2["base_runs"][0]["correct"]   # only ~8 pts better
    assert SC.score(R2)["gates"]["controls_clearly_worse(>=10pt,lo>0)"] is False

def test_both_controls_needed(tmp):
    fb = fb_block(ctl=0.3); fb["teach"]["empty_correct"] = fb["teach"]["real_correct"]     # empty control as good as real
    R = synth(tmp, feedback=fb); R["feedback"]["exam_after_correct"] = R["base_runs"][0]["correct"]
    assert SC.score(R)["gates"]["controls_clearly_worse(>=10pt,lo>0)"] is False

def test_cluster_ci_wider_than_item_ci():
    g = np.random.default_rng(0); cl = np.repeat(np.arange(8), 6); eff = (g.random(8) < 0.5)[cl].astype(float)
    m, lo, hi = SC.paired_boot(eff); m2, lo2, hi2 = SC.paired_boot(eff, clusters=cl)
    assert (hi2 - lo2) > (hi - lo) * 1.3

def test_crossed_bootstrap_runs():
    cl = np.tile(np.arange(10), 3); dr = np.repeat(np.arange(3), 10); d = np.random.default_rng(1).random(30) < .5
    m, lo, hi = SC.paired_boot(d.astype(float), clusters=cl, draws=dr); assert lo <= m <= hi

def test_rule8_per_cell_ceiling(tmp):
    R = synth(tmp, base_acc=1.0); o = SC.score(R); assert "base_above_chance_below_ceiling_every_cell" in o["failed_gates"]

def test_cell_size_and_ood_kinds_gates(tmp):
    R = synth(tmp); keep = [i for i, (s, k) in enumerate(zip(R["source"], R["kind"])) if not (s == "S2" and k == "nocall")]
    for key in ("cand_correct", "kind", "source"): R[key] = [R[key][i] for i in keep]
    for r in R["base_runs"]: r["correct"] = [r["correct"][i] for i in keep]
    o = SC.score(R); assert "ood_block_has_every_S1_kind" in o["failed_gates"]

def test_retention_conditioning_unit():
    b = np.array([1, 1, 1, 0, 0, 1], bool); c = np.array([1, 0, 1, 1, 1, 1], bool); r = SC.retention(b, c); assert (r["n_base_correct"], r["kept"]) == (4, 3)
    assert SC.retention(np.zeros(5, bool), np.ones(5, bool))["ret"] == 0.0
