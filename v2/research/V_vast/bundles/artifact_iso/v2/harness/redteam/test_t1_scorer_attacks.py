"""Adversarial inputs for tournament/T1/scorer.py. Each case states the CORRECT credit. `orig` = scorer as shipped, `hard` = scorer_hardened.py.
test_orig_* failures are FINDINGS (wrong credit in the shipped scorer). test_hard_* must all pass (proves the proposed fix) and
test_hard_regression_* proves the fix does not break gold-vs-gold on real non-locked rows and quantifies flips on stored base outputs."""
import json, os, sys, collections, pytest
from conftest import T1, V2, load, D1
import scorer_v1 as orig   # the original shipped scorer (kept as tournament/T1/scorer_v1.py)
import scorer as hard       # ADOPTED hardened scorer (tournament/T1/scorer.py)

TC = lambda o: "<tool_call>" + o + "</tool_call>"
def tool(name, **types): return dict(name=name, description="", parameters=dict(type="object", properties={k: dict(type=v) for k, v in types.items()}))
def item(gold, tools=None, alts=False): return dict(gold=gold, gold_alts=alts, tools=tools or [])
G = lambda n, a: [dict(name=n, arguments=a)]

# (id, model_text, item, correct_credit)
CASES = [
 ("bigint_collision", TC('{"name":"f","arguments":{"id":12345678901234568}}'), item(G("f", {"id": 12345678901234567}), [tool("f", id="integer")]), False),
 ("leading_zero_zip_str", TC('{"name":"f","arguments":{"z":"2134"}}'), item(G("f", {"z": "02134"}), [tool("f", z="string")]), False),
 ("str_true_for_string_param", TC('{"name":"f","arguments":{"b":"True"}}'), item(G("f", {"b": "true"}), [tool("f", b="string")]), False),
 ("str_for_bool_param_ok", TC('{"name":"f","arguments":{"b":"true"}}'), item(G("f", {"b": True}), [tool("f", b="boolean")]), True),
 ("str_num_for_string_param", TC('{"name":"f","arguments":{"c":"5"}}'), item(G("f", {"c": 5}), [tool("f", c="string")]), False),
 ("str_num_for_int_param_ok", TC('{"name":"f","arguments":{"c":"5"}}'), item(G("f", {"c": 5}), [tool("f", c="integer")]), True),
 ("inner_ws_collapsed", TC('{"name":"f","arguments":{"c":"a  b"}}'), item(G("f", {"c": "a b"}), [tool("f", c="string")]), False),
 ("arabic_indic_digit", TC('{"name":"f","arguments":{"c":"\\u0663"}}'), item(G("f", {"c": 3}), [tool("f", c="integer")]), False),
 ("str_1.0_for_int_param", TC('{"name":"f","arguments":{"c":"1.0"}}'), item(G("f", {"c": 1}), [tool("f", c="string")]), False),
 ("alts_true_vs_1", TC('{"name":"f","arguments":{"c":true}}'), item(G("f", {"c": [1]}), [tool("f", c="integer")], True), False),
 ("alts_1_vs_true", TC('{"name":"f","arguments":{"c":1}}'), item(G("f", {"c": [True]}), [tool("f", c="boolean")], True), False),
 ("alts_str_for_int_bfcl_strict", TC('{"name":"f","arguments":{"c":"10"}}'), item(G("f", {"c": [10]}), [tool("f", c="integer")], True), True),  # we accept (documented leniency); BFCL itself would not
 ("nocall_pycall_syntax", 'get_weather(city="Paris")', item([], [tool("get_weather", city="string")]), False),
 ("nocall_quoted_example", 'I cannot. An example would be {"name": "get_weather", "arguments": {"city": "x"}}', item([], [tool("get_weather", city="string")]), False),  # ambiguous: full call-shaped JSON in prose counts as a call (conservative, both scorers)
 ("nocall_prose_name_dict", 'Record: {"name": "Alice"}', item([], [tool("get_weather", city="string")]), True),
 ("think_block_mentions_call", '<think>I could <tool_call>{"name":"f","arguments":{}}</tool_call></think>No tool fits.', item([], [tool("f")]), True),
 ("think_block_then_real_call", '<think>maybe <tool_call>{"name":"f","arguments":{}}</tool_call></think>\n<tool_call>{"name":"f","arguments":{}}</tool_call>', item(G("f", {}), [tool("f")]), True),
 ("truncated_unclosed_tag", '<tool_call>{"name":"f","arguments":{}}', item(G("f", {}), [tool("f")]), False),   # hit max_tokens: strict reviewers call this a failure
 ("dup_gold_one_call", TC('{"name":"f","arguments":{"a":1}}'), item(G("f", {"a": 1}) * 2, [tool("f", a="integer")]), False),
 ("dup_gold_two_calls", TC('{"name":"f","arguments":{"a":1}}') * 2, item(G("f", {"a": 1}) * 2, [tool("f", a="integer")]), True),
 ("extra_halluc_call_on_call_item", TC('{"name":"f","arguments":{"a":1}}') + TC('{"name":"g","arguments":{}}'), item(G("f", {"a": 1}), [tool("f", a="integer")]), False),
 ("nocall_explain_then_call", "Sure! " + TC('{"name":"f","arguments":{}}'), item([], [tool("f")]), False),
 ("two_tags_one_bad", TC('{"name":"f","arguments":{}}') + TC("{bad"), item(G("f", {}), [tool("f")]), False),
 ("null_for_missing", TC('{"name":"f","arguments":{"a":null}}'), item(G("f", {}), [tool("f", a="string")]), False),
 ("empty_string_for_missing", TC('{"name":"f","arguments":{"a":""}}'), item(G("f", {}), [tool("f", a="string")]), False),
 ("name_case", TC('{"name":"F","arguments":{}}'), item(G("f", {}), [tool("f")]), False),
 ("nested_key_order_ok", TC('{"arguments":{"a":{"y":1,"x":2}},"name":"f"}'), item(G("f", {"a": {"x": 2, "y": 1}}), [tool("f", a="object")]), True),
 ("multi_order_ok", TC('{"name":"g","arguments":{}}') + TC('{"name":"f","arguments":{}}'), item(G("f", {}) + G("g", {}), [tool("f"), tool("g")]), True),
 ("args_as_json_string_ok", TC('{"name":"f","arguments":"{\\"a\\":1}"}'), item(G("f", {"a": 1}), [tool("f", a="integer")]), True),
 ("float_near_miss", TC('{"name":"f","arguments":{"a":0.1000001}}'), item(G("f", {"a": 0.1}), [tool("f", a="number")]), False),
 ("int_float_equal_ok", TC('{"name":"f","arguments":{"a":2}}'), item(G("f", {"a": 2.0}), [tool("f", a="number")]), True),
 ("exact_ok", TC('{"name":"f","arguments":{"a":"x y"}}'), item(G("f", {"a": "x y"}), [tool("f", a="string")]), True),
 ("list_order_matters_unordered_param", TC('{"name":"f","arguments":{"a":[2,1]}}'), item(G("f", {"a": [1, 2]}), [tool("f", a="array")]), False),  # strict by design; documented false-negative, symmetric across base/cand
 ("nocall_refusal_ok", "I can't help with that using these tools.", item([], [tool("f")]), True),
 ("empty_output_on_nocall", "", item([], [tool("f")]), False),   # POLICY (adopted 2026-11-06): empty/EOS/whitespace is not an abstention
 ("empty_output_on_call", "", item(G("f", {}), [tool("f")]), False),
]
IDS = [c[0] for c in CASES]
KNOWN_ORIG_BUGS = {"bigint_collision", "leading_zero_zip_str", "str_true_for_string_param", "str_num_for_string_param", "inner_ws_collapsed", "arabic_indic_digit",
                   "str_1.0_for_int_param", "alts_true_vs_1", "alts_1_vs_true", "nocall_pycall_syntax", "think_block_mentions_call", "think_block_then_real_call",
                   "truncated_unclosed_tag", "nocall_prose_name_dict"}
# nocall_quoted_example / nocall_prose_name_dict: orig already returns False(=wrong) for the first -> bug in the OTHER direction (see below)
ORIG_WRONG_DIRECTION = set()

@pytest.mark.parametrize("cid,text,it,want", CASES, ids=IDS)
def test_orig(cid, text, it, want):
    got = orig.score_item(text, it)["correct"]
    if cid in KNOWN_ORIG_BUGS | ORIG_WRONG_DIRECTION | {"empty_output_on_nocall"} and got != want: pytest.xfail(f"FINDING shipped scorer gives credit={got}, should be {want}")
    assert got == want

@pytest.mark.parametrize("cid,text,it,want", CASES, ids=IDS)
def test_hard(cid, text, it, want):
    assert hard.score_item(text, it)["correct"] == want

def test_summary_count_orig_bugs(capsys):
    bad = [c[0] for c in CASES if orig.score_item(c[1], c[2])["correct"] != c[3]]
    good = [c[0] for c in CASES if hard.score_item(c[1], c[2])["correct"] != c[3]]
    print("ORIG wrong credit on:", bad); print("HARD wrong credit on:", good)
    assert not good

def rows(split): return load(f"{D1}/{split}.jsonl")

def test_hard_regression_gold_vs_gold_nonlocked():
    bad = 0; n = 0
    for split in ["s1_dev", "s3_dev"]:
        for r in rows(split)[:1500]:
            if r["gold"] and any(not isinstance(v, (int, float, str, bool, list, dict, type(None))) for g in r["gold"] for v in g["arguments"].values()): continue
            for style in ["qwen", "llama"]:
                n += 1; bad += not hard.score_item(orig.gold_text(r, style), r)["correct"]
    print("hardened gold-vs-gold failures:", bad, "/", n)
    assert bad / n < 0.005

def test_rescore_stored_outputs_orig_vs_hard(capsys):
    """Impact on REAL base outputs: rescore tournament/T1/results/*.json rows (vLLM bf16 outputs on 600 dev probe items) with both scorers."""
    items = {r["id"]: r for r in load(f"{T1}/probe_items.jsonl")}
    tot = collections.Counter()
    for f in sorted(os.listdir(f"{T1}/results")):
        if not f.endswith(".json") or f == "SUMMARY.json": continue
        d = json.load(open(f"{T1}/results/{f}")); flips = []
        for r in d["rows"]:
            it = items[r["id"]]; a = orig.score_item(r["output"], it)["correct"]; b = hard.score_item(r["output"], it)["correct"]
            if a != b: flips.append((r["id"], r["kind"], a, b))
        print(f"{f}: {len(d['rows'])} rows, orig!=hard on {len(flips)}", flips[:6]); tot[f] = len(flips)
    assert all(v <= 30 for v in tot.values())

def test_empty_output_on_nocall_is_credited_by_v1_only(capsys):
    """v1 policy leak (kept as a finding about the old scorer): empty / whitespace / EOS earned full credit on every nocall item. The ADOPTED scorer rejects it."""
    for t in ["", "\n\n  ", "<|im_end|>", "<|im_end|><|endoftext|><|endoftext|>", "<think>\n\n</think>\n\n", "..."]:
        assert orig.score_item(t, item([], [tool("f")]))["correct"] is True or t == "..."
        r = hard.score_item(t, item([], [tool("f")])); assert r["correct"] is False and r["empty"] is True, t
    assert hard.score_item("No tool fits.<|im_end|><|endoftext|>", item([], [tool("f")]))["correct"] is True

def test_pycall_false_positive_fixed():
    """review fix: prose 'search (the web)' on a tool named search was a false parse failure in scorer_hardened.py"""
    assert hard.score_item("I could use search (the web) but nothing fits.", item([], [tool("search", q="string")]))["correct"] is True
    assert hard.score_item("search(q='x')", item([], [tool("search", q="string")]))["correct"] is False
    assert hard.score_item("Available Symbols Endpoint()", item([], [tool("Available Symbols Endpoint")]))["correct"] is False

def test_think_variants():
    t = item(G("f", {}), [tool("f")])
    assert hard.score_item('reasoning...</think>\n<tool_call>{"name":"f","arguments":{}}</tool_call>', t)["correct"]     # opening tag lived in the prompt
    assert not hard.score_item('<think>still thinking <tool_call>{"name":"f","arguments":{}}</tool_call>', t)["correct"]   # truncated in thinking
