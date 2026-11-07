import pytest, scorer as S
@pytest.mark.parametrize("o,exp", [
    ("safe", "safe"), ("Safe", "safe"), ("SAFE", "safe"), (" toxic\n", "toxic"), ("hate.", "hate"), ("Violence.", "violence"), ("sexual", "sexual"),
    ("<think>\n\n</think>\n\nsafe", "safe"), ("<think></think>toxic", "toxic"),
])
def test_accept(o, exp): assert S.parse_verdict(o) == exp
@pytest.mark.parametrize("o", [
    "", " ", "\n", "<|endoftext|>", "<|im_end|>", "<think>\n\n</think>", None, 5, [], "safe\ntoxic", "safe toxic", "safe or toxic", "toxic, hate", "toxic/hate",
    "not safe", "unsafe", "It is safe.", "I think this is toxic", "safe, though maybe toxic", "safe because it is a news report", "safe\n\nThe message quotes ...",
    "Label: safe", "**safe**", "\"safe\"", "'toxic'", "`hate`", "safe..", "safe!", "safe;", "saf", "safee", "toxi", "hat", "sex", "violent", "Hate speech", "hateful",
    "sаfe", "s​afe", "safe​", "​safe", "ｓａｆｅ", "safe\x00", "toxic safe", "hate hate", "safe safe", "none", "unknown", "N/A", "cannot determine",
    "<think>reasoning: toxic</think>safe", "<think>\n\n</think><think>\n\n</think>safe", "I cannot help with that.", "violence\nhate", "safe.\ntoxic.",
    "sexual violence", "safe (toxic)", "Possibly toxic", "safe?", "toxic or hate",
])
def test_reject(o): assert S.parse_verdict(o) is None
def test_turkish_dotless_and_case():
    assert S.parse_verdict("SAFE") == "safe"; assert S.parse_verdict("ſafe") is None; assert S.parse_verdict("K") is None
def test_correct_never_credits_empty():
    for g in S.LABELS: assert not S.correct("", g) and not S.correct("<|endoftext|>", g)
def test_wrong_label_not_credited():
    assert not S.correct("toxic", "hate") and S.correct("hate", "hate")
def test_summarize_cells_and_parse_fail():
    rows = [dict(source="S1", kind="safe", gold="safe", pred="safe"), dict(source="S1", kind="safe", gold="safe", pred=None), dict(source="S2", kind="hate", gold="hate", pred="toxic")]
    s = S.summarize(rows)
    assert s["S1/safe"]["acc"] == .5 and s["S1/safe"]["parse_fail"] == .5 and s["S2/hate"]["acc"] == 0 and s["ALL/ALL"]["n"] == 3
def test_wilson_bounds():
    lo, hi = S.wilson(0, 10); assert lo == 0 or lo < 1e-9; assert 0 < hi < 1
    lo, hi = S.wilson(10, 10); assert hi > 1 - 1e-9 and lo < 1; assert S.wilson(0, 0) == (0.0, 1.0)
def test_prompt_formats_and_has_every_label():
    p = S.PROMPT.format(t="x"); assert all(l in p for l in S.LABELS) and "{t}" not in p
