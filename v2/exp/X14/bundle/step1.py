"""X14 step 1: guard models vs generic Qwen3-8B on the same 300 dev items. Real generation, logprobs for margins."""
import json, re, gc
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
items = json.load(open("data/items300.json"))
MODELS = [("qwen3guard", "Qwen/Qwen3Guard-Gen-8B"), ("llamaguard3", "VincentGOURBIN/Llama-Guard-3-8B"), ("qwen3_generic", "Qwen/Qwen3-8B")]
VERD = {"safe": "Safe", "unsafe": "Unsafe", "controversial": "Controversial", "toxic": "Unsafe"}
def build(name, tok, it):
    if name == "qwen3_generic":
        return tok.apply_chat_template([{"role": "user", "content": it["generic_user"]}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
    msgs = [{"role": "user", "content": it["prompt"]}] + ([{"role": "assistant", "content": it["response"]}] if it["response"] else [])
    return tok.apply_chat_template(msgs, tokenize=False).strip() if name == "llamaguard3" else tok.apply_chat_template(msgs, tokenize=False)
def verdict(text):
    m = re.search(r"\b(unsafe|safe|controversial|toxic)\b", text.lower())
    return VERD[m.group(1)] if m else None
res = {}
for name, mid in MODELS:
    tok = AutoTokenizer.from_pretrained(mid)
    llm = LLM(mid, dtype="bfloat16", max_model_len=4096, gpu_memory_utilization=0.85, seed=0)
    prompts = [build(name, tok, it) for it in items]
    print(name, "SAMPLE PROMPT:", prompts[0][-400:].replace("\n", "\\n"), flush=True)
    sp = SamplingParams(temperature=0, max_tokens=24 if name != "qwen3_generic" else 8, logprobs=5)
    inp = prompts if name != "llamaguard3" else [{"prompt_token_ids": tok(p, add_special_tokens=False).input_ids} for p in prompts]
    outs = llm.generate(inp, sp, use_tqdm=False)
    rows = []
    for it, o in zip(items, outs):
        c = o.outputs[0]; toks = []
        for t, lp in zip(c.token_ids, c.logprobs):
            vals = sorted(((v.logprob, v.decoded_token) for v in lp.values()), reverse=True)
            toks.append((vals[0][1], vals[0][0] - vals[1][0] if len(vals) > 1 else 99.0))
        vi = next((i for i, (t, m) in enumerate(toks) if t.strip().lower() in VERD), None)
        le = vi + 1 if vi is not None else len(toks)
        rows.append(dict(id=it["id"], task=it["task"], src=it["src"], gold=it["gold"], text=c.text, verdict=verdict(c.text),
                         vmargin=toks[vi][1] if vi is not None else None, minmargin_line=min(m for _, m in toks[:le]) if toks else None, minmargin_all=min(m for _, m in toks) if toks else None))
    res[name] = rows
    json.dump(res, open("results/step1_raw.json", "w"))
    del llm; gc.collect()
    import torch; torch.cuda.empty_cache()
