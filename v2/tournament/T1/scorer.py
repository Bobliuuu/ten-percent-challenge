"""T1 scorer v2 (hardened; adopted 2026-11-06 from harness/redteam/scorer_hardened.py after review). scorer_v1.py = the original, kept for audit (its parser is copied verbatim here). Interface: score_item(text, item) -> dict(correct, parse_fail, n_calls, pred, empty); gold_text(item, style).
Changes vs v1 (each has a test in harness/redteam/test_t1_scorer_attacks.py and tournament/T1/test_scorer.py):
  1. numbers canonicalised with Decimal (no float collapse of 17-digit ids); bool is NOT a number; "1" vs True never equal.
  2. string<->number / string<->bool coercion ONLY when the item's tool schema declares that param number/integer/boolean AND the string is canonical ASCII.
  3. inner whitespace is NOT collapsed (strip + NFC only).
  4. <think>...</think> removed before parsing (also an unclosed <think> and a leading '...</think>' when the opening tag was in the prompt).
  5. bare {"name": X} without an arguments/parameters key is not a call outside <tool_call> tags.
  6. python-style call syntax naming a declared tool (name(a=1) / name()) is an ATTEMPTED CALL: wrong on nocall (parse_fail), never credited.
  7. unterminated <tool_call> (truncation) is a parse failure.
  8. EMPTY output (nothing but special tokens/whitespace/punctuation: fewer than 2 alphanumerics after removing special tokens) is NOT an abstention: wrong on every kind.
Review fixes vs scorer_hardened.py: (a) PYCALL matched only [A-Za-z_][\\w.]* identifiers followed by optional whitespace + '(' -> prose like "search (the web)" on a tool named search
  was a false parse failure and tool names with spaces were missed; now exact tool-name + '(' with no space and an '=' or ')' as first non-blank content. (b) BFCL-alts path ignored
  the declared parameter type (string "true" for a boolean param was rejected there but accepted elsewhere); now typed. (c) unclosed <think> / leading '</think>' handled. (d) empty-output rule (8)."""
import json, re, unicodedata
from decimal import Decimal
_dec = json.JSONDecoder()
# ---- v1 tag/JSON parser and gold renderer, copied verbatim from scorer_v1.py so that scorer.py is a single self-contained file (Modal images copy only scorer.py)
def _calls_from_obj(o, out):
    if isinstance(o, list):
        for x in o: _calls_from_obj(x, out)
    elif isinstance(o, dict):
        if "name" in o and ("arguments" in o or "parameters" in o):
            a = o.get("arguments", o.get("parameters"))
            if isinstance(a, str):
                try: a = json.loads(a)
                except Exception: raise ValueError("args str")
            if not isinstance(a, dict): raise ValueError("args type")
            out.append((str(o["name"]), a))
        elif "name" in o and len(o) == 1: out.append((str(o["name"]), {}))
        elif "function" in o and isinstance(o["function"], dict): _calls_from_obj(o["function"], out)

def parse_calls(text):
    """returns (calls[(name,args)], parse_fail:bool)"""
    calls, fail = [], False
    # gpt-oss harmony
    for m in re.finditer(r"to=functions\.([^<]+?)(?:\s+json)?\s*(?:<\|constrain\|>\w*)?<\|message\|>(.*?)(?:<\|call\|>|<\|end\|>|<\|return\|>|$)", text, re.S):
        body = m.group(2).strip()
        try: a = json.loads(body) if body else {}
        except Exception: fail = True; continue
        calls.append((m.group(1), a if isinstance(a, dict) else {}))
    if calls or "to=functions." in text: return calls, fail or not calls
    # qwen-style tags
    tags = re.findall(r"<tool_call>(.*?)(?:</tool_call>|$)", text, re.S)
    if tags:
        for t in tags:
            try: _calls_from_obj(json.loads(t.strip()), calls)
            except Exception: fail = True
        return calls, fail or not calls
    # bare json dicts / lists anywhere (llama-3.1, untagged)
    t = text.replace("<|python_tag|>", "")
    i = 0
    while i < len(t):
        if t[i] in "[{":
            try:
                o, j = _dec.raw_decode(t, i); tmp = []
                try: _calls_from_obj(o, tmp)
                except ValueError: fail = True
                if tmp: calls += tmp; i = j; continue
            except Exception: pass
        i += 1
    if not calls and re.search(r'"(name|function)"\s*:', t) and re.search(r'"(arguments|parameters)"\s*:', t): fail = True
    return calls, fail


def gold_text(item, style="qwen"):
    """render gold as model text (for unit tests / sanity)"""
    if style == "qwen": return "\n".join("<tool_call>\n" + json.dumps(dict(name=g["name"], arguments=g["arguments"])) + "\n</tool_call>" for g in item["gold"]) or "I cannot help with that."
    if style == "llama": return "; ".join(json.dumps(dict(name=g["name"], parameters=g["arguments"])) for g in item["gold"]) or "None of the tools apply."
    if style == "harmony": return "".join(f"<|channel|>commentary to=functions.{g['name']} <|constrain|>json<|message|>{json.dumps(g['arguments'])}<|call|>" for g in item["gold"]) or "<|channel|>final<|message|>No tool fits.<|return|>"


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

SPECIAL = re.compile(r"<\|[^|<>]*\|>|</?s>|<think>\s*</think>")
def is_empty(text):
    t = SPECIAL.sub("", text)
    return len(re.findall(r"[^\W_]", t)) < 2

def strip_think(text):
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    if "<think>" in text: text = text.split("<think>")[0]            # truncated inside the thinking block: nothing after it counts
    elif "</think>" in text: text = text.split("</think>", 1)[1]      # opening tag was part of the prompt
    return text

def attempted_pycall(text, item):
    for t in item.get("tools", []):
        if re.search(r"(?<![\w.])" + re.escape(t["name"]) + r"\((?:[^()]*=|\s*\))", text): return True
    return False

def parse(text, item):
    text = strip_think(text)
    if "<tool_call>" in text and not re.search(r"</tool_call>", text.split("<tool_call>")[-1]): return [], True
    calls, fail = parse_calls(text)
    if "<tool_call>" not in text and "to=functions." not in text and calls and not re.search(r'"(arguments|parameters)"\s*:', text):
        calls = []   # bare {"name": X} with no arguments key is prose/data, not a call
    if not calls and not fail and attempted_pycall(text, item): return [], True
    return calls, fail

def score_item(text, item):
    gold = item["gold"]
    if is_empty(text): return dict(correct=False, parse_fail=False, n_calls=0, pred=[], empty=True)
    calls, fail = parse(text, item)
    pred = [dict(name=n, arguments=a) for n, a in calls]
    if fail: return dict(correct=False, parse_fail=True, n_calls=len(calls), pred=pred, empty=False)
    if not gold: return dict(correct=len(calls) == 0, parse_fail=False, n_calls=len(calls), pred=pred, empty=False)
    if len(calls) != len(gold): return dict(correct=False, parse_fail=False, n_calls=len(calls), pred=pred, empty=False)
    if item.get("gold_alts"):
        def ok(a, g, pt):
            for k, v in a.items():
                if k not in g: return False
                if not any(tag(v, pt.get(k)) == tag(x, pt.get(k)) or (x != "" and tag(v, "number") == tag(x, "number") and not isinstance(v, bool) and not isinstance(x, bool)) for x in g[k] if x != "" or v == ""): return False
            return all(k in a or "" in al for k, al in g.items())
        def bt(i, used):
            if i == len(calls): return True
            n, a = calls[i]; pt = ptypes(item, n)
            return any(gi not in used and g["name"] == n and ok(a, g["arguments"], pt) and bt(i + 1, used | {gi}) for gi, g in enumerate(gold))
        return dict(correct=bt(0, frozenset()), parse_fail=False, n_calls=len(calls), pred=pred, empty=False)
    a = sorted(map(repr, (ckey(item, n, x) for n, x in calls))); b = sorted(map(repr, (ckey(item, g["name"], g["arguments"]) for g in gold)))
    return dict(correct=a == b, parse_fail=False, n_calls=len(calls), pred=pred, empty=False)
