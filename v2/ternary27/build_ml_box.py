"""Build multilingual + code proxy items (public HF datasets; runs on the box). Output ml_items.jsonl, ml_meta.json (tests)."""
import json, random, collections, os
os.environ["HF_HOME"] = "/home/ubuntu/work/hf"
from datasets import load_dataset
R = random.Random(0); TOXP = open("/home/ubuntu/work/tox_prompt.txt").read(); items = []; meta = {}
# MGSM (11 languages x 250)
for lang in ["en", "es", "fr", "de", "ru", "zh", "ja", "th", "sw", "bn", "te"]:
    ds = load_dataset("juletxara/mgsm", lang, split="test")
    for i, r in enumerate(ds):
        items.append(dict(id=f"mgsm|{lang}|{i}", task="mgsm", source=lang, kind="math", gold=str(r["answer_number"]), user=r["question"] + "\n\nSolve step by step, then give the final answer on the last line as 'Answer: <number>'.", maxnew=600))
# multilingual toxicity (binary), 300 per language
ds = load_dataset("textdetox/multilingual_toxicity_dataset")
for lang in ds.keys():
    rows = list(ds[lang]); R.shuffle(rows); pos = [r for r in rows if r["toxic"] == 1][:150]; neg = [r for r in rows if r["toxic"] == 0][:150]
    for i, r in enumerate(pos + neg):
        items.append(dict(id=f"mltox|{lang}|{i}", task="mltox", source=lang, kind="toxic" if r["toxic"] else "safe", gold="toxic" if r["toxic"] else "safe", user=TOXP.replace("{t}", r["text"]), maxnew=16))
# code
for r in load_dataset("google-research-datasets/mbpp", "sanitized", split="test"):
    items.append(dict(id=f"mbpp|{r['task_id']}", task="code", source="mbpp", kind="py", gold="", user=r["prompt"] + "\nYour code should pass these tests:\n" + "\n".join(r["test_list"]) + "\nReturn only the complete Python function(s) in one ```python block.", maxnew=512))
    meta[f"mbpp|{r['task_id']}"] = dict(kind="mbpp", tests=r["test_list"], imports=r.get("test_imports", []))
for r in load_dataset("openai/openai_humaneval", split="test"):
    items.append(dict(id=f"he|{r['task_id']}", task="code", source="humaneval", kind="py", gold="", user="Complete the following Python function. Return the complete function in one ```python block.\n\n" + r["prompt"], maxnew=512))
    meta[f"he|{r['task_id']}"] = dict(kind="he", test=r["test"], entry=r["entry_point"], prompt=r["prompt"])
with open("/home/ubuntu/work/ml_items.jsonl", "w") as f:
    for x in items: f.write(json.dumps(x, ensure_ascii=False) + "\n")
json.dump(meta, open("/home/ubuntu/work/ml_meta.json", "w"))
print(collections.Counter((x["task"], x["source"]) for x in items))
