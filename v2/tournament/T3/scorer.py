"""T3 scorer. Task format (fixed, documented): model answers a chat prompt with ONE word, "human" or "ai" (case-insensitive,
optional quotes / markdown / trailing punctuation, e.g. `AI`, `"human".`, `**ai**`). Optional reasoning blocks (<think>...</think>,
gpt-oss harmony analysis channel) are stripped before parsing. Anything else (empty, other word, both labels in the first
two words such as "human or ai", a refusal) is a FORMAT FAILURE and counts as INCORRECT (never as a guess).
Item correct <=> parsed verdict == gold ('human'|'ai'). Kinds = source_gold (e.g. S1_human, S2_ai)."""
import re, collections

def strip_reasoning(s):
    if s is None: return ""
    if "<|channel|>final<|message|>" in s: s = s.split("<|channel|>final<|message|>")[-1]
    s = re.sub(r"<\|[a-z_]+\|>", " ", s) if "<|" in s else s
    s = re.sub(r"<think>.*?</think>", "", s, flags=re.S | re.I)
    if "</think>" in s: s = s.split("</think>")[-1]
    return s.strip()

def parse_verdict(out):
    """-> 'human' | 'ai' | None (format failure)."""
    s = strip_reasoning(out)
    lines = [l for l in s.splitlines() if l.strip()]
    if not lines: return None
    toks = re.findall(r"[a-z]+", lines[0].lower())
    if not toks or toks[0] not in ("human", "ai"): return None
    if len(toks) > 1 and toks[1] in ("or", "and", "vs", "versus"): return None
    return toks[0]

def correct(out, gold): return parse_verdict(out) == gold

def summarize(rows):
    """rows: dicts with keys out, gold, kind, source. -> dict of accuracy overall / per kind / per source, format-failure rate, pred dist."""
    res = {"n": len(rows)}; byk = collections.defaultdict(list); bys = collections.defaultdict(list); fail = 0; pd = collections.Counter()
    for r in rows:
        v = parse_verdict(r["out"]); c = v == r["gold"]; fail += v is None; pd[v or "FAIL"] += 1
        byk[r["kind"]].append(c); bys[r["source"]].append(c)
    m = lambda l: sum(l) / len(l) if l else float("nan")
    res.update(acc=m([c for v in byk.values() for c in v]), format_fail=fail / max(1, len(rows)), pred_dist=dict(pd),
               per_kind={k: m(v) for k, v in sorted(byk.items())}, per_source={k: m(v) for k, v in sorted(bys.items())})
    return res
