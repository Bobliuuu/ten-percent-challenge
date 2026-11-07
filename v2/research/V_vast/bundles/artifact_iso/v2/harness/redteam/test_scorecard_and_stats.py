"""Attacks on harness/scorecard.py (gates) and on the statistics: Wilson bounds with n, power, noise floor from two independent bf16 runs."""
import json, math, os, collections
import numpy as np, pytest
import scorecard as SC
from conftest import T1, V2, load

_TREE = {}
def _tree():
    """artifact tree used by the synthetic candidates: model.safetensors + config.json, allowed manifest lists both; one grep needle."""
    import tempfile
    if "d" not in _TREE:
        d = tempfile.mkdtemp(); open(f"{d}/model.safetensors", "wb").write(os.urandom(4096)); open(f"{d}/config.json", "w").write("{}"); _TREE["d"] = d
    return _TREE["d"]

def synth(n_s1=450, n_s2=None, ret_cells=None, base_acc=0.8, seed=0, kinds=("nocall", "single", "multi"), extra=False):
    """locked-like arrays with the v2 exam shape (cells >= 330). n_s1 = items per S1 kind. ret_cells: {(source,kind): retention} applied to base-correct items."""
    g = np.random.default_rng(seed); src, kind, b, c = [], [], [], []
    sizes = {("S1", k): n_s1 for k in kinds} | {("S2", "single"): 500, ("S2", "multi"): 372, ("S2", "nocall"): 400}
    for (s, k), m in sizes.items():
        if k not in kinds: continue
        for _ in range(m):
            bi = g.random() < base_acc; ci = bi and (g.random() < (ret_cells or {}).get((s, k), 1.0)); src.append(s); kind.append(k); b.append(bool(bi)); c.append(bool(ci))
    runs = [dict(correct=b, engine="vllm", batch=32), dict(correct=b, engine="hf", batch=8), dict(correct=b, engine="vllm", batch=1)]
    man = {"model.safetensors": None, "config.json": None} | ({} if not extra else {})
    d = _tree()
    return dict(base_runs=runs, cand_correct=c, kind=kind, source=src, size=dict(cand_bytes=9.5, base_fp16_bytes=100), chance=0.1, generative=True, kinds_real_output=True,
                artifact_tree=dict(dir=d, allowed_manifest=({"config.json": None} if extra else man), needles=["a needle that is not in the artifact tree"]), base_params=8e9)

def test_baseline_perfect_candidate_passes_exam_gates():
    o = SC.score(synth()); fails = [k for k in o["failed_gates"] if k not in ("feedback_present",)]
    assert fails == [] , fails

def test_FINDING_ood_source_cell_can_collapse_and_all_exam_gates_pass():
    """S2/nocall retention 0.88 (300 items) while S1 is perfect: kind aggregate and overall still clear the gates. Rule 2 says 'at least 95% within
    every kind of item (each answer class, SOURCE or category)': a source x kind cell < 95% is a violation under the strict reading."""
    o = SC.score(synth(ret_cells={("S2", "nocall"): 0.88}))
    exam_gates = {k: v for k, v in o["gates"].items() if k != "feedback_present"}
    print("failed:", o["failed_gates"], "| per-kind:", {k: round(v["ret"], 3) for k, v in o["details"]["retention_per_kind"].items()})
    assert not all(exam_gates.values()), "scorecard passes a candidate with 88% retention on S2/nocall"   # EXPECTED TO FAIL

def test_FINDING_ood_source_retention_not_gated():
    o = SC.score(synth(ret_cells={("S2", "single"): 0.90, ("S2", "multi"): 0.90, ("S2", "nocall"): 0.90}))
    print("overall", o["details"]["retention_overall"]["ret"], "gates failed", o["failed_gates"])
    # S2 at 90%, S1 100% -> overall about 0.95: caught by overall; make S2 a smaller share to show the hole
    o2 = SC.score(synth(n_s1=6000, ret_cells={("S2", "single"): 0.92, ("S2", "multi"): 0.92, ("S2", "nocall"): 0.92}))
    print("with 6000 S1:", o2["details"]["retention_overall"]["ret"], o2["failed_gates"])
    assert any("ood" in g or "source" in g for g in o2["failed_gates"]), "no gate looks at retention within the OOD source"   # EXPECTED TO FAIL

def test_control_gate_matches_redesign():
    """REDESIGN.md section 4: real - control >= 10 points with CI lower bound > 0, for EVERY control (empty, shuffled/mixed-label, random-wrong-tool)."""
    src = open(f"{V2}/harness/scorecard.py").read()
    assert "controls_clearly_worse(>=10pt,lo>0)" in src and 'v["mean"] >= 0.10 and v["lo"] > 0' in src and "controls_complete" in src

def test_no_new_parts_computed_from_tree_not_declared():
    """v2: extra_files / grep_train_hits are computed from the artifact tree; a self-declared clean record is ignored and cannot make the gate pass."""
    R = synth(); R["no_new_parts"] = dict(extra_files=[], grep_train_hits=0); assert SC.score(R)["gates"]["no_new_parts"] is True       # tree really is clean
    R = synth(extra=True); R["no_new_parts"] = dict(extra_files=[], grep_train_hits=0); o = SC.score(R)                                    # tree has a file the manifest does not allow
    assert o["gates"]["no_new_parts"] is False and "model.safetensors" in o["details"]["artifact_tree_audit"]["extra_files"]
    R = synth(); del R["artifact_tree"]; R["no_new_parts"] = dict(extra_files=[], grep_train_hits=0); assert SC.score(R)["gates"]["no_new_parts"] is False  # declared-only fails closed

def test_wilson_per_kind_needs_what_point_retention():
    """With n locked per kind and base accuracy p, how high must observed retention be for the Wilson LOWER bound to clear 0.95 / 0.98?"""
    rows = []
    for n_locked in (300, 437, 700, 1000):
        for p in (0.6, 0.8, 0.9):
            nb = int(n_locked * p)
            for target in (0.95, 0.98):
                need = next(k for k in range(nb + 1) if SC.wilson_lo(k, nb) >= target) if SC.wilson_lo(nb, nb) >= target else None
                rows.append((n_locked, p, nb, target, None if need is None else round(need / nb, 4)))
    for r in rows: print("n_locked=%d base_acc=%.1f n_base_correct=%d target_lo=%.2f -> observed retention must be >= %s" % r)
    # headline: 300 locked items, base .6 -> 180 base-correct: need >=~98.9% observed to certify 95% with 95% confidence
    nb = 180; k = next(k for k in range(nb + 1) if SC.wilson_lo(k, nb) >= 0.95)
    assert k / nb > 0.975

def test_wilson_overall_vs_98_with_2637_items():
    nb = int(2637 * 0.78); k = next(k for k in range(nb + 1) if SC.wilson_lo(k, nb) >= 0.98)
    print("overall n_base_correct", nb, "need observed retention", round(k / nb, 4), "to certify >=98% with 95% confidence")
    assert k / nb > 0.985

def test_paired_boot_ignores_clustering():
    """teach/neighbor items generated from a handful of templates are not independent. v2: paired_boot accepts cluster ids (tool family / template) and the CI widens to the cluster bootstrap."""
    g = np.random.default_rng(0); n_clusters, per = 8, 6   # 48 teach items from 8 templates
    cl = np.repeat(np.arange(n_clusters), per); eff = g.random(n_clusters) < 0.5   # edit works on ~half of templates, all-or-nothing
    d = eff[cl].astype(float) * 1.0
    m0, lo0, hi0 = SC.paired_boot(d)                      # plain item bootstrap: still too narrow (documented)
    m, lo, hi = SC.paired_boot(d, clusters=cl)            # v2 cluster bootstrap
    bs = [d[np.concatenate([np.where(cl == c)[0] for c in g.integers(0, n_clusters, n_clusters)])].mean() for _ in range(2000)]
    clo, chi = np.quantile(bs, [.025, .975])
    print(f"item bootstrap CI [{lo0:.2f},{hi0:.2f}] vs scorecard cluster CI [{lo:.2f},{hi:.2f}] vs reference cluster bootstrap [{clo:.2f},{chi:.2f}] (mean {m:.2f})")
    assert (hi - lo) >= (chi - clo) * 0.9 and (hi - lo) > (hi0 - lo0)

# ---------------------------------------------------------------- noise floor from REAL data: two independent bf16 runs of the same model on the same 600 items
def test_base_vs_base_noise_floor_two_engines(capsys):
    import scorer
    items = {r["id"]: r for r in load(f"{T1}/probe_items.jsonl")}
    out = {}
    for m in ["Qwen3-8B", "Qwen3-14B"]:
        a = json.load(open(f"{T1}/results/{m}.json"))["rows"]                               # vLLM 0.11, bf16
        bpath = f"{V2}/compress/dense/results/t1/t1_{m}_bf16.json"
        if not os.path.exists(bpath): continue
        b = {r["id"]: r for r in json.load(open(bpath))["t1"]}                              # HF/dense.py bf16
        ids = [r["id"] for r in a]
        ca = np.array([r["correct"] for r in a], bool); cb = np.array([str(b[i]["correct"]) == "True" for i in ids], bool)
        kind = np.array([r["kind"] for r in a]); src = np.array([r["source"] for r in a])
        same_text = np.mean([a[j]["output"].strip() == b[ids[j]]["output"].strip() for j in range(len(ids))])
        ov = SC.retention(ca, cb); ov2 = SC.retention(cb, ca)
        pk = {k: round(SC.retention(ca, cb, kind == k)["ret"], 3) for k in ["nocall", "single", "multi"]}
        ps = {s: round(SC.retention(ca, cb, src == s)["ret"], 3) for s in ["S1", "S3"]}
        out[m] = dict(acc_vllm=ca.mean(), acc_hf=cb.mean(), identical_text=same_text, ret_overall=ov["ret"], ret_overall_rev=ov2["ret"], per_kind=pk, per_src=ps, n_bc=ov["n_base_correct"])
        print(m, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in out[m].items()})
    assert out, "no paired bf16 runs found"
    for m, v in out.items():
        assert v["ret_overall"] >= 0.98, f"{m}: the SAME bf16 model run on a different engine keeps only {v['ret_overall']:.3f} of its own correct items (98% rule is inside the noise)"

def test_noise_floor_k_run_protocol_documented():
    """Fix: define 'original' as the MAJORITY/INTERSECTION of K>=3 base runs (different batch sizes/engines), and report retention(base_run_i -> base_run_j) as the
    floor beside every candidate number. This test asserts the helper exists in the harness (it does not today)."""
    import importlib
    try: importlib.import_module("noise_floor")
    except ImportError: pytest.xfail("no noise_floor helper in harness; see redteam/noise_floor.py (proposed)")


def test_FINDING_rule8_ceiling_per_kind_on_probe_data(capsys):
    """Rule 8: 'scores well above chance and below its ceiling'. scorecard only checks the OVERALL base accuracy (<0.97). Per (source, kind) base accuracy from the stored vLLM probes:"""
    out = {}
    for m in ["Qwen3-8B", "Qwen3-14B", "Qwen3-30B-A3B-Instruct-2507"]:
        rows = json.load(open(f"{T1}/results/{m}.json"))["rows"]; c = collections.defaultdict(list)
        for r in rows: c[(r["source"], r["kind"])].append(r["correct"])
        out[m] = {f"{k[0]}/{k[1]}": round(float(np.mean(v)), 3) for k, v in c.items()}
        print(m, out[m])
    worst = max(v for m in out.values() for k, v in m.items() if k.endswith("nocall"))
    assert worst < 0.97, f"base is at ceiling (acc {worst}) on S1/nocall: retention on that kind is not informative and a reviewer will cite rule 8"   # EXPECTED TO FAIL

def test_retention_conditioning_unit():
    b = np.array([1, 1, 1, 0, 0, 1], bool); c = np.array([1, 0, 1, 1, 1, 1], bool)
    r = SC.retention(b, c)
    assert (r["n_base_correct"], r["kept"]) == (4, 3)          # candidate wins on base-wrong items do NOT offset losses
    assert SC.retention(np.zeros(5, bool), np.ones(5, bool))["ret"] == 0.0   # fails closed when base has nothing right

def test_parse_failure_counts_as_loss():
    import scorer
    it = dict(gold=[dict(name="f", arguments={})], tools=[])
    assert scorer.score_item("<tool_call>{bad", it) ["correct"] is False


def test_scorecard_rule8_per_cell_gate_exists():
    """the per-(source,kind) ceiling/chance gate exists in the scorecard (the legacy-probe finding above stays red until the new hard nocall is re-probed with a model)."""
    o = SC.score(synth(base_acc=1.0)); assert "base_above_chance_below_ceiling_every_cell" in o["failed_gates"]
