"""Strict TOOLSEL scorer. correct iff the model's decoded output, after removing chat special tokens (<|im_end|>, <|endoftext|>) and an EMPTY think block
(<think>\n\n</think>), equals the gold answer exactly after .strip() of surrounding whitespace. Nothing else is credited:
empty / EOS-only output, two names, a name plus explanation, quotes, backticks, trailing period, different case, prefix/suffix of the name, `None` for `none`, a tool call JSON."""
import re
SPECIAL = ("<|im_end|>", "<|endoftext|>", "<|im_start|>")
def clean(text):
    t = text
    for s in SPECIAL: t = t.replace(s, "")
    t = re.sub(r"<think>\s*</think>", "", t)
    return t.strip()
def score_toolsel(text, item):
    t = clean(text)
    return dict(correct=bool(t) and t == item["answer"], empty=not t, pred=t)
if __name__ == "__main__":
    it = dict(answer="get_weather"); n = dict(answer="none")
    cases = [("get_weather", it, 1), ("get_weather<|im_end|>", it, 1), ("  get_weather\n", it, 1), ("<think>\n\n</think>\n\nget_weather", it, 1), ("none", n, 1), ("", it, 0), ("<|im_end|>", n, 0), ("None", n, 0), ("none.", n, 0), ("`get_weather`", it, 0), ('"get_weather"', it, 0),
             ("get_weather\nget_time", it, 0), ("get_weather, get_time", it, 0), ("get_weather()", it, 0), ("The tool is get_weather", it, 0), ("get_weath", it, 0), ("get_weather_v2", it, 0), ('<tool_call>{"name":"get_weather","arguments":{}}</tool_call>', it, 0), ("GET_WEATHER", it, 0), ("none\nnone", n, 0)]
    bad = [(t, e) for t, i, e in cases if score_toolsel(t, i)["correct"] != bool(e)]
    print("scorer tests", len(cases), "failures", bad); assert not bad
