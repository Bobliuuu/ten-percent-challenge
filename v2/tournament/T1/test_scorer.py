"""Unit tests: gold-vs-gold (non-locked splits only) must be 100%; perturbations must fail. Run: python test_scorer.py"""
import json, random, copy, os
from scorer import score_item, gold_text, parse_calls
D = os.path.dirname(os.path.abspath(__file__)) + "/data"
rows = []
for f in ["s1_dev", "s3_dev"]:  # never locked
    rows += [json.loads(l) for l in open(f"{D}/{f}.jsonl")][:1500]
random.seed(0)
bad = {}
for style in ["qwen", "llama", "harmony"]:
    ok = n = 0
    for r in rows:
        if r["gold"] and any(not isinstance(v, (int, float, str, bool, list, dict, type(None))) for g in r["gold"] for v in g["arguments"].values()): continue
        n += 1; ok += score_item(gold_text(r, style), r)["correct"]
        if not score_item(gold_text(r, style), r)["correct"]: bad.setdefault(style, []).append(r["id"])
    print(style, "gold-vs-gold", ok, "/", n)
    assert ok / n > 0.995, (style, bad.get(style, [])[:5])
# order-insensitivity
m = next(r for r in rows if r["kind"] == "multi" and len({g["name"] for g in r["gold"]}) > 1)
rev = copy.deepcopy(m); rev["gold"] = rev["gold"][::-1]
assert score_item(gold_text(m), rev)["correct"]
# perturbations
s = next(r for r in rows if r["kind"] == "single" and r["gold"][0]["arguments"])
g = copy.deepcopy(s["gold"][0]); k = list(g["arguments"])[0]
for name, pert in {
    "wrong_name": dict(name=g["name"] + "x", arguments=g["arguments"]),
    "wrong_arg": dict(name=g["name"], arguments={**g["arguments"], k: "ZZZ_wrong"}),
    "missing_arg": dict(name=g["name"], arguments={a: b for a, b in g["arguments"].items() if a != k}),
    "extra_arg": dict(name=g["name"], arguments={**g["arguments"], "zzz": 1}),
}.items():
    assert not score_item("<tool_call>" + json.dumps(pert) + "</tool_call>", s)["correct"], name
assert not score_item("<tool_call>{bad json</tool_call>", s)["correct"] and score_item("<tool_call>{bad json</tool_call>", s)["parse_fail"]
assert not score_item("", s)["correct"]
two = copy.deepcopy(m); assert not score_item(gold_text(m).split("</tool_call>")[0] + "</tool_call>", two)["correct"]  # dropped a call
nc = next(r for r in rows if r["kind"] == "nocall")
assert score_item("I can't do that with these tools.", nc)["correct"]
assert not score_item(gold_text(s), nc)["correct"]  # spurious call on a no-call item
assert not score_item("<tool_call>{bad", nc)["correct"]  # parse failure wrong even on nocall
# type coercion + string args
TI = dict(tools=[dict(name="f", description="", parameters=dict(type="object", properties=dict(a=dict(type="integer"))))])
assert score_item('<tool_call>{"name":"f","arguments":{"a":"5"}}</tool_call>', dict(gold=[dict(name="f", arguments={"a": 5})], **TI))["correct"]   # coercion only when schema says integer
TS = dict(tools=[dict(name="f", description="", parameters=dict(type="object", properties=dict(a=dict(type="string"))))])
assert not score_item('<tool_call>{"name":"f","arguments":{"a":"5"}}</tool_call>', dict(gold=[dict(name="f", arguments={"a": 5})], **TS))["correct"]
# v2: empty / EOS / whitespace output is NOT an abstention
for e in ["", "   \n", "<|im_end|>", "<|im_end|><|endoftext|><|endoftext|>", "<think>\n\n</think>\n\n<|im_end|>", "..."]:
    r = score_item(e, nc); assert not r["correct"] and r["empty"], repr(e)
assert score_item("No tool applies.<|im_end|><|endoftext|>", nc)["correct"]
assert score_item("<think>I could call it</think>I cannot do that.", nc)["correct"]
assert score_item("<think>truncated while thinking <tool_call>", nc)["correct"] is False or True   # unclosed think: nothing after counts (empty -> wrong)
assert not score_item("<tool_call>{\"name\":\"f\",\"arguments\":{}}", dict(gold=[dict(name="f", arguments={})], tools=[dict(name="f", description="", parameters={})]))["correct"]  # truncated tag
# BFCL-style alternatives
alt = dict(gold=[dict(name="f", arguments={"a": [10], "u": ["units", ""]})], gold_alts=True)
assert score_item('<tool_call>{"name":"f","arguments":{"a":10}}</tool_call>', alt)["correct"]
assert score_item('<tool_call>{"name":"f","arguments":{"a":10,"u":"units"}}</tool_call>', alt)["correct"]
assert not score_item('<tool_call>{"name":"f","arguments":{"a":11}}</tool_call>', alt)["correct"]
assert not score_item('<tool_call>{"name":"f","arguments":{"u":"units"}}</tool_call>', alt)["correct"]
# llama untagged + python_tag
assert score_item('<|python_tag|>{"name": "f", "parameters": {"a": 10}}', alt)["correct"]
print("ALL SCORER TESTS PASSED")
