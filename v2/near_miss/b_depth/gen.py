"""gen.py MODELDIR OUT.json [KEEPIDS.json] : real greedy vLLM generation on the dev eval items (T1 600 + 1400 cls). Same items/order as native_survey2/box/probe_box.py."""
import os, json, sys, random, time
D = "/code/data"
model, out = sys.argv[1], sys.argv[2]; keep = json.load(open(sys.argv[3])) if len(sys.argv) > 3 else None
from vllm import LLM, SamplingParams
t1 = [json.loads(l) for l in open(D + "/probe_items.jsonl")]
TOXP = open(D + "/tox_prompt.txt").read(); INJP = open(D + "/inj_prompt.txt").read()
cls = []
for f in ("tox_s1_dev.jsonl", "tox_s3_dev.jsonl"): cls += [("tox", json.loads(l)) for l in open(D + "/" + f)]
for f in ("inj_dev_S1.jsonl", "inj_dev_S3.jsonl"): cls += [("inj", json.loads(l)) for l in open(D + "/" + f)]
random.Random(0).shuffle(cls); cls = cls[:1400]
easy = [json.loads(l) for l in open(D + "/e_easy.jsonl")]
llm = LLM(model, dtype="bfloat16", max_model_len=6144, gpu_memory_utilization=0.9, enforce_eager=True, max_num_seqs=256)
tok = llm.get_tokenizer()
def tpl(msgs, tools=None):
    try: return tok.apply_chat_template(msgs, tools=tools, add_generation_prompt=True, tokenize=False, enable_thinking=False)
    except TypeError: return tok.apply_chat_template(msgs, tools=tools, add_generation_prompt=True, tokenize=False)
p1 = [tpl([{"role": "user", "content": it["query"]}], [{"type": "function", "function": t} for t in it["tools"]]) for it in t1]
pe = [tpl([{"role": "user", "content": r["user"]}]) for r in easy]
p2 = [tpl([{"role": "user", "content": (TOXP if k == "tox" else INJP).replace("{t}", r["text"])}]) for k, r in cls]
kw = dict(allowed_token_ids=keep) if keep else {}
# OOV rate of eval prompts under the kept vocab
oov = None
if keep:
    ks = set(keep); tot = bad = 0; ex = 0
    for p in p1 + p2 + pe:
        ids = tok(p, add_special_tokens=False)["input_ids"]; tot += len(ids); b = sum(1 for i in ids if i not in ks); bad += b; ex += b > 0
    oov = dict(tok_oov=bad / tot, prompts_with_oov=ex / len(p1 + p2 + pe))
t0 = time.time()
o1 = llm.generate(p1, SamplingParams(temperature=0, max_tokens=300, **kw)); o2 = llm.generate(p2, SamplingParams(temperature=0, max_tokens=16, **kw))
oe = llm.generate(pe, SamplingParams(temperature=0, max_tokens=8, **kw))
R = {"easy": [{"id": r["id"], "kind": r["kind"], "source": r["src"], "gold": r["answer"], "out": o.outputs[0].text} for r, o in zip(easy, oe)], "model": model, "oov": oov, "t1": [{"id": it["id"], "kind": it["kind"], "source": it["source"], "gold": it["gold"], "out": o.outputs[0].text} for it, o in zip(t1, o1)],
     "cls": [{"task": k, "id": r["id"], "gold": r["gold"], "kind": r.get("kind"), "source": r.get("source"), "out": o.outputs[0].text} for (k, r), o in zip(cls, o2)], "secs": time.time() - t0}
json.dump(R, open(out, "w")); print("GEN_DONE", out, R["secs"], oov, flush=True)
