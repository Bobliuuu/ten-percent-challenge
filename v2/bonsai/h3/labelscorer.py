"""Label scorer for the generative label tasks: the whole generated output must be the gold label word (case-insensitive, optional single '.').
Row format (harness v3): gold = [{"name": <label>, "arguments": {}}]."""
import re
def score_item(text, item):
    g = item["gold"][0]["name"]; s = re.sub(r"^\s*<think>\s*</think>\s*", "", text or "").strip()
    ok = re.fullmatch(re.escape(g) + r"\.?", s, re.I) is not None
    return dict(correct=ok, parse_fail=not re.fullmatch(r"[A-Za-z_]+\.?", s), n_calls=1)
