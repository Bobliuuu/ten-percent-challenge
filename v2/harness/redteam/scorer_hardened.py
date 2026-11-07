"""Proposed hardened T1 scorer (red-team patch). Same interface as tournament/T1/scorer.py: score_item(text, item).
Changes vs original (each backed by a test in test_t1_scorer_attacks.py):
  1. numbers canonicalised with Decimal (no float collapse of 17-digit ids); bool is NOT a number; "1" vs True never equal.
  2. string<->number / string<->bool coercion ONLY when the item's tool schema declares that param number/integer/boolean AND the string is canonical ASCII
     (no leading zeros, no unicode digits). Otherwise strings compare as strings.
  3. inner whitespace is NOT collapsed (strip + NFC only).
  4. <think>...</think> removed before parsing.
  5. bare {"name": X} without an arguments/parameters key is not a call outside <tool_call> tags.
  6. python-style call syntax naming a declared tool (f(a=1)) is an ATTEMPTED CALL: wrong on nocall (parse_fail), never credited.
  7. unterminated <tool_call> (truncation) is a parse failure.
"""
import json, re, unicodedata
from decimal import Decimal
import scorer_v1 as base

def _dec_loads(s): return json.loads(s, parse_float=Decimal, parse_int=int)

def tag(v, ptype=None):
    if isinstance(v, bool): return ("b", v)
    if v is None: return ("z",)
    if isinstance(v, (int, Decimal)): return ("n", str(Decimal(v).normalize()))
    if isinstance(v, float): return ("n", str(Decimal(repr(v)).normalize()))
    if isinstance(v, str):
        s = unicodedata.normalize("NFC", v.strip())
        if ptype in ("integer", "number") and re.fullmatch(r"-?(0|[1-9][0-9]*)(\.[0-9]+)?", s, re.A): return ("n", str(Decimal(s).normalize()))
        if ptype == "boolean" and s in ("true", "false"): return ("b", s == "true")
        return ("s", s)
    if isinstance(v, (list, tuple)): return ("l", tuple(tag(x, None) for x in v))
    if isinstance(v, dict): return ("d", tuple(sorted((str(k), tag(x, None)) for k, x in v.items())))
    return ("?", repr(v))

def ptypes(item, name):
    for t in item.get("tools", []):
        if t["name"] == name: return {k: p.get("type") for k, p in (t.get("parameters", {}).get("properties") or {}).items()}
    return {}

def ckey(item, name, args):
    pt = ptypes(item, name)
    return (name, tuple(sorted((k, tag(v, pt.get(k))) for k, v in args.items())))

PYCALL = re.compile(r"(?<![\w.])([A-Za-z_][\w.]*)\s*\(")

def parse(text, item):
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    if "<tool_call>" in text and not re.search(r"</tool_call>", text.split("<tool_call>")[-1]): return [], True
    calls, fail = base.parse_calls(text)
    if "<tool_call>" not in text and "to=functions." not in text and calls and not re.search(r'"(arguments|parameters)"\s*:', text):
        calls = []   # bare {"name": X} with no arguments key is prose/data, not a call
    if not calls and not fail:
        names = {t["name"] for t in item.get("tools", [])}
        if any(m.group(1) in names for m in PYCALL.finditer(text)): return [], True   # attempted call in unsupported syntax
    return calls, fail

def score_item(text, item):
    calls, fail = parse(text, item); gold = item["gold"]
    pred = [dict(name=n, arguments=a) for n, a in calls]
    if fail: return dict(correct=False, parse_fail=True, n_calls=len(calls), pred=pred)
    if not gold: return dict(correct=len(calls) == 0, parse_fail=False, n_calls=len(calls), pred=pred)
    if len(calls) != len(gold): return dict(correct=False, parse_fail=False, n_calls=len(calls), pred=pred)
    if item.get("gold_alts"):
        def ok(a, g):
            for k, v in a.items():
                if k not in g: return False
                if not any(tag(v) == tag(x) or (x != "" and tag(v, "number") == tag(x, "number") and not isinstance(v, bool) and not isinstance(x, bool)) for x in g[k] if x != "" or v == ""): return False
            return all(k in a or "" in al for k, al in g.items())
        def bt(i, used):
            if i == len(calls): return True
            n, a = calls[i]
            return any(gi not in used and g["name"] == n and ok(a, g["arguments"]) and bt(i + 1, used | {gi}) for gi, g in enumerate(gold))
        return dict(correct=bt(0, frozenset()), parse_fail=False, n_calls=len(calls), pred=pred)
    a = sorted(map(repr, (ckey(item, n, x) for n, x in calls))); b = sorted(map(repr, (ckey(item, g["name"], g["arguments"]) for g in gold)))
    return dict(correct=a == b, parse_fail=False, n_calls=len(calls), pred=pred)
