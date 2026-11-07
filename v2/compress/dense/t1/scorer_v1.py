"""T1 scorer: parse generated text into tool calls and exact-match against gold.
Correct iff: gold empty -> model emitted NO tool call and no parse failure;
             gold non-empty -> multiset of (name, canonical args) equal (order-insensitive), no parse failure.
Parse failure (a tool-call marker present but unparseable JSON) => wrong for every kind.
Canonicalisation: strings stripped + whitespace-collapsed (case-SENSITIVE); numeric strings and numbers compare equal as floats;
bool stays bool; lists ordered; dicts key-sorted. Gold with gold_alts (BFCL S2): each arg maps to a list of allowed values;
'' in the list = param may be omitted; a model arg must equal one allowed value; missing arg ok only if '' allowed.
Parsers: Qwen <tool_call>{json}</tool_call>, gpt-oss harmony `to=functions.NAME ... <|message|>{json}`, Llama-3.1/bare JSON dicts
{"name":..,"parameters"|"arguments":..}, and list-of-dicts. Same parser for every base."""
import json, re
_dec = json.JSONDecoder()

def _num(s):
    try: return float(s)
    except Exception: return None

def canon(v):
    if isinstance(v, bool) or v is None: return v
    if isinstance(v, (int, float)): return float(v)
    if isinstance(v, str):
        s = re.sub(r"\s+", " ", v.strip())
        if re.fullmatch(r"-?\d+(\.\d+)?", s): return float(s)
        if s.lower() in ("true", "false"): return s.lower() == "true"
        return s
    if isinstance(v, (list, tuple)): return [canon(x) for x in v]
    if isinstance(v, dict): return {str(k): canon(x) for k, x in sorted(v.items())}
    return v

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

def _key(name, args): return (name, json.dumps(canon(args), sort_keys=True))

def _match_alts(model_args, gold_args):
    for k, v in model_args.items():
        if k not in gold_args: return False
        al = gold_args[k]
        if not any(canon(v) == canon(x) for x in al if x != "" or v == ""): return False
    for k, al in gold_args.items():
        if k not in model_args and "" not in al: return False
    return True

def score_item(text, item):
    """-> dict(correct, parse_fail, n_calls, pred)"""
    calls, fail = parse_calls(text)
    gold = item["gold"]
    pred = [dict(name=n, arguments=a) for n, a in calls]
    if fail: return dict(correct=False, parse_fail=True, n_calls=len(calls), pred=pred)
    if not gold: return dict(correct=len(calls) == 0, parse_fail=False, n_calls=len(calls), pred=pred)
    if len(calls) != len(gold): return dict(correct=False, parse_fail=False, n_calls=len(calls), pred=pred)
    if item.get("gold_alts"):
        left = list(range(len(gold))); ok = True
        for n, a in calls:  # greedy bipartite is fine: names + alternatives rarely collide; use backtracking for safety
            pass
        def bt(ci, used):
            if ci == len(calls): return True
            n, a = calls[ci]
            for gi, g in enumerate(gold):
                if gi not in used and g["name"] == n and _match_alts(a, g["arguments"]) and bt(ci + 1, used | {gi}): return True
            return False
        return dict(correct=bt(0, frozenset()), parse_fail=False, n_calls=len(calls), pred=pred)
    a = sorted(_key(n, x) for n, x in calls); b = sorted(_key(g["name"], g["arguments"]) for g in gold)
    return dict(correct=a == b, parse_fail=False, n_calls=len(calls), pred=pred)

def gold_text(item, style="qwen"):
    """render gold as model text (for unit tests / sanity)"""
    if style == "qwen": return "\n".join("<tool_call>\n" + json.dumps(dict(name=g["name"], arguments=g["arguments"])) + "\n</tool_call>" for g in item["gold"]) or "I cannot help with that."
    if style == "llama": return "; ".join(json.dumps(dict(name=g["name"], parameters=g["arguments"])) for g in item["gold"]) or "None of the tools apply."
    if style == "harmony": return "".join(f"<|channel|>commentary to=functions.{g['name']} <|constrain|>json<|message|>{json.dumps(g['arguments'])}<|call|>" for g in item["gold"]) or "<|channel|>final<|message|>No tool fits.<|return|>"
