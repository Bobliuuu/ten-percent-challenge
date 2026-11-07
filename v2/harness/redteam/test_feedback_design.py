"""Adversarial checks of the feedback method (feedback_v3: lm_head covariance-whitened row edit / late-layer down_proj MEMIT edit) using the STORED results
and the code. CPU only, no model. Teach/neighbor construction is re-derived from non-locked dev data."""
import json, os, re, glob, hashlib, collections
import numpy as np, pytest
import scorecard as SC
from conftest import V2, D1, load, norm

FB = f"{V2}/feedback_v3"
h8 = lambda s: int(hashlib.sha256(s.encode()).hexdigest()[:8], 16)

def fb4_src(): return open(f"{FB}/fb4.py").read()
def fb3_src(): return open(f"{FB}/fb3.py").read()

# ---------------------------------------------------------------- construction audits
def test_fb3_neighbors_assumed_base_correct_not_verified():
    s = fb3_src()
    assert "base_n=np.ones" in s and "base_nd=np.ones" in s
    pytest.xfail("fb3 neighbours: 'items the base gets right' are ASSUMED (np.ones), never filtered by an actual base run; 'neighbor retention' is then mixed with base errors. fb4 neighbour pools: see next test")

def test_twin_tool_is_a_duplicate_schema_with_suffix():
    """The T1 'mistake' = a synthetic twin: copy of the gold tool with name+'_v2' inserted into the tool list and declared correct. Description+params identical:
    nothing in the prompt distinguishes the twin; the 'right' answer is an arbitrary preference."""
    s = fb4_src(); assert 'twin=dict(gt[0],name=al)' in s and '"_v2"' in s
    pytest.xfail("FINDING: confusable-pair feedback is a naming preference on a cloned schema, not an error the model made through a missing capability")

def test_T1pairS_shuffled_control_is_complement_not_permutation():
    s = fb4_src()
    seg = s[s.index("def shuf(anns,seed):\n                out=[]"):][:600]
    assert 'replace("_v2","")' in seg
    # all annotations share one label (-> the twin). Any permutation of labels is identical to the real labels, so the only informative shuffle is the complement, which
    # restates the base's default behaviour (base picks the original ~77-87% of time) => 'shuffled' ~ 'empty feedback' and cannot show label-specificity.
    pytest.xfail("FINDING: shuffled control for the twin scenario is the inverse labelling = base default; use a mixed-label (50% twin / 50% original) and a random-wrong-tool control")

def _split_stats():
    raw = load(f"{D1}/s1_dev.jsonl") + load(f"{D1}/s3_dev.jsonl")
    singles = [x for x in raw if x["kind"] == "single" and len(x["gold"]) == 1 and len(x["tools"]) <= 6]
    sp = {"ann": [], "dev": [], "teach": [], "neigh": []}
    for x in singles:
        hh = h8(x["id"]) % 10
        sp["neigh" if hh == 9 else "ann" if hh < 4 else "dev" if hh < 6 else "teach"].append(x)
    return sp

def test_teach_items_reuse_annotation_tools_and_queries(capsys):
    sp = _split_stats(); ann = sp["ann"][:400]; teach = sp["teach"][:60]
    gold_names = {x["gold"][0]["name"] for x in ann}; tools_ann = {t["name"] for x in ann for t in x["tools"]}
    f_gold = np.mean([x["gold"][0]["name"] in gold_names for x in teach])
    f_tool = np.mean([all(t["name"] in tools_ann for t in x["tools"]) for x in teach])
    aq = [set(norm(x["query"]).split()) for x in ann]
    f_near = np.mean([any(len(set(norm(x["query"]).split()) & a) / len(set(norm(x["query"]).split()) | a) >= 0.6 for a in aq) for x in teach])
    ng = {x["gold"][0]["name"] for x in sp["neigh"]}; f_ng = np.mean([x["gold"][0]["name"] in gold_names for x in sp["neigh"]])
    print(f"teach items whose gold base-tool also appears as an annotation gold: {f_gold:.2f}; all tools seen in ann: {f_tool:.2f}; J>=.6 query neighbour in ann: {f_near:.2f}; neighbour items whose gold tool was annotated: {f_ng:.2f}")
    # hostile reading: teach-set is 'same tools, same generator' -> measures memorised-tool transfer, not feedback generalisation to unseen tools/styles
    assert f_gold < 0.5, f"{f_gold:.0%} of teach items are about a tool that was in the annotation set"   # EXPECTED TO FAIL

def test_neighbors_are_mostly_the_annotated_tools(capsys):
    sp = _split_stats(); gold_ann = {x["gold"][0]["name"] for x in sp["ann"][:400]}
    n_overlap = np.mean([x["gold"][0]["name"] in gold_ann for x in sp["neigh"][:408]])
    print("neighbour pool items whose gold tool is an annotated tool:", round(float(n_overlap), 3))
    # rule 5: 'nearby exam items the feedback could affect' must be about the SAME tools (twin of the same base tool). Neighbours that never touch annotated tools prove nothing.
    assert n_overlap > 0.5

# ---------------------------------------------------------------- statistics on stored results
def _load_results(name): return json.load(open(f"{FB}/results/{name}.json"))

@pytest.mark.parametrize("name", ["Qwen3-8B_T1pairS_first", "Qwen3-14B_T1pairS_first"])
def test_seed_pseudoreplication_in_teach_CI(name, capsys):
    R = _load_results(name); bt = np.array(R["base_teach"], float)
    F = [x for x in R["final"] if x["method"] == "lmhead_lda"]
    stack = np.stack([np.array(x["real"]["teach"], float) for x in F])   # seeds x items
    concat = np.concatenate([s - bt for s in stack]); m1, lo1, hi1 = SC.paired_boot(concat)
    per_item = (stack - bt).mean(0); m2, lo2, hi2 = SC.paired_boot(per_item)   # n = teach items (60), seeds averaged
    # annotation-draw variance too: bootstrap over (seed) is only 3 draws -> report range
    rng = [float((s - bt).mean()) for s in stack]
    print(f"{name}: harness CI (180 pseudo-items) [{lo1:+.3f},{hi1:+.3f}] vs item-level n={len(bt)} [{lo2:+.3f},{hi2:+.3f}]; per-seed gains {np.round(rng,3).tolist()}")
    assert (hi2 - lo2) > (hi1 - lo1), "harness CI is NARROWER than the item-level CI (pseudo-replication)"   # passes if the finding holds

def test_new_tool_scenario_fails_G2():
    """T1new (teach a tool the model has never seen). This is what the maintainer's own 'different distribution' feedback most plausibly looks like."""
    R = _load_results("Qwen3-8B_T1new_first"); out = {}
    bt = np.array(R["base_teach"], float)
    for meth in ["lmhead_lda", "blend", "memit_down"]:
        F = [x for x in R["final"] if x["method"] == meth]
        gain = np.mean([np.mean(x["real"]["teach"]) for x in F]) - bt.mean(); neigh = np.mean([np.mean(x["real"]["neigh"]) for x in F])
        out[meth] = (round(float(gain), 3), round(float(neigh), 3))
    print("T1new (gain, neighbours):", out)
    assert any(g >= 0.15 and n >= 0.98 for g, n in out.values()), "no method passes G2 on the new-tool scenario"   # EXPECTED TO FAIL

def test_lmhead_and_memit_G2_summary_all_scenarios(capsys):
    rows = json.load(open(f"{FB}/table.json"))
    print("G2 passes by (task, method):", collections.Counter((r["task"], r["method"], r["G2"]) for r in rows if r.get("G2") is not None))

def test_neighbor_ci_with_n100():
    """Neighbour gate is retention >= 0.98 on 100-200 items: 1 broken item of 100 is 0.99 but the Wilson lower bound is ~0.95."""
    for n in (100, 200, 408):
        k = n - 1
        print(f"n={n}: observed {k/n:.3f}, Wilson lo {SC.wilson_lo(k, n):.3f}; need 0 breaks for lo>=0.96: {SC.wilson_lo(n, n):.3f}")
    assert SC.wilson_lo(99, 100) < 0.98 and SC.wilson_lo(100, 100) < 0.98   # n=100 can never certify >=98% at 95% confidence

# ---------------------------------------------------------------- mathematical equivalence: the lm_head edit IS a few-shot LDA discriminant
def test_lmhead_lda_edit_is_a_fitted_linear_classifier():
    """Reproduce fb3.lmhead (mode='lda'): row[t] += alpha*||row||*d, d = Sigma^-1 (mean_h(sites) - mu_gen) / norm. logit_t gain on hidden h is alpha*||row|| * d.h = a
    scaled LDA discriminant between 'annotation contexts' and 'generic contexts'. Show: (1) it ranks held-out contexts of the annotated class above generic ones (AUC>0.95)
    -> it behaves as a classifier fitted to the annotations; (2) it fires on any context with large d.h (neighbour damage) in proportion to anisotropy."""
    rng = np.random.default_rng(0); D = 64
    A = rng.normal(size=(D, D)); Sig = A @ A.T / D + 0.05 * np.eye(D)                    # generic anisotropic hidden covariance
    mu_gen = rng.normal(size=D); L = np.linalg.cholesky(Sig)
    gen = mu_gen + rng.normal(size=(5000, D)) @ L.T
    shift = rng.normal(size=D) * 0.8                                                     # annotated class has a mean shift
    ann = mu_gen + shift + rng.normal(size=(25, D)) @ L.T                                # 25 annotations
    held = mu_gen + shift + rng.normal(size=(500, D)) @ L.T
    d = np.linalg.solve(Sig + 1e-2 * np.diag(Sig).mean() * np.eye(D), ann.mean(0) - mu_gen); d /= np.linalg.norm(d)
    s_held, s_gen = held @ d, gen @ d
    auc = np.mean(s_held[:, None] > s_gen[None, :500])
    print("AUC held-out annotated-class vs generic contexts under the edit direction:", round(float(auc), 3))
    assert auc > 0.9    # passes: the edit direction is a working few-shot linear classifier
    frac_gen_fired = np.mean(s_gen > np.quantile(s_held, 0.5)); print("generic contexts scoring above median held-out:", round(float(frac_gen_fired), 3))

# ---------------------------------------------------------------- prompt-leak guard (reusable by the final harness)
def assert_no_feedback_in_prompt(prompt_before: str, prompt_after: str, annotations: list):
    assert prompt_before == prompt_after, "prompt changed after feedback was applied"
    for a in annotations:
        for field in ("query", "gold", "wrong"):
            v = a.get(field)
            if isinstance(v, str) and len(v) > 12: assert v not in prompt_after, f"annotation {field} leaked into prompt"

def test_prompt_leak_guard_detects_leak():
    assert_no_feedback_in_prompt("sys: tools...\nuser: hi", "sys: tools...\nuser: hi", [dict(query="a long annotated query text", gold="foo")])
    with pytest.raises(AssertionError): assert_no_feedback_in_prompt("p", "p + a long annotated query text", [dict(query="a long annotated query text")])
    with pytest.raises(AssertionError): assert_no_feedback_in_prompt("p", "q", [])

def test_fb_code_does_not_touch_prompt_after_edit():
    """static: prompts are built by chat()/P() before any edit and the edit functions only mutate weights (no string concatenation of annotation text into prompt)."""
    s = fb4_src()
    apply_fn = s[s.index("def apply(cfg,anns)"):][:900]
    assert "chat(" not in apply_fn and "prompt" not in apply_fn.replace("prompts", "")

def test_edit_has_no_backward_or_optimizer():
    for src in (fb3_src(), fb4_src()):
        code = "\n".join(l for l in src.split("\n") if not l.strip().startswith("#"))
        assert not re.search(r"\.backward\(|optim\.|loss\.backward|torch\.optim", code)
    # no_grad scale search is 'forward-only' but IS a fitted hyper-parameter search per annotation draw
    assert "lam" in fb4_src()
