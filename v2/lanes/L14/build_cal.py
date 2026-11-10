"""L14: broad unlabeled calibration text (general chat/QA, math, code, wikitext). No labels, no task text, no Banking77, no exam data.
Output data/cal_general.jsonl rows {id, src, user, resp?}. Seeded; re-runnable."""
import json, random, urllib.request, io, re, sys
import pandas as pd
R = random.Random(14)
def get(u): return urllib.request.urlopen(urllib.request.Request(u, headers={"User-Agent": "Mozilla/5.0"}), timeout=120).read()
rows = []
def add(src, user, resp=None):
    user = user.strip()
    if len(user) < 15: return
    r = dict(id=f"L14|{src}|{len(rows)}", src=src, user=user[:1500])
    if resp: r["resp"] = resp.strip()[:700]
    rows.append(r)
# chat / QA
d = [json.loads(l) for l in get("https://huggingface.co/datasets/databricks/databricks-dolly-15k/resolve/main/databricks-dolly-15k.jsonl").decode().splitlines()]
R.shuffle(d); n = 0
for x in d:
    if len(x["instruction"]) + len(x["context"]) > 1200 or len(x["response"]) < 20: continue
    add("dolly", x["instruction"] + (("\n\n" + x["context"]) if x["context"] else ""), x["response"]); n += 1
    if n >= 700: break
# math
g = [json.loads(l) for l in get("https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/train.jsonl").decode().splitlines()]
R.shuffle(g)
for x in g[:400]: add("gsm8k", x["question"], re.sub(r"<<.*?>>", "", x["answer"]))
# code
c = json.loads(get("https://huggingface.co/datasets/sahil2801/CodeAlpaca-20k/resolve/main/code_alpaca_20k.json")); R.shuffle(c); n = 0
for x in c:
    if len(x["output"]) < 20: continue
    add("code", x["instruction"] + (("\n\n" + x["input"]) if x["input"] else ""), x["output"]); n += 1
    if n >= 400: break
# wikitext paragraphs (raw text as the user turn, no response)
df = pd.read_parquet(io.BytesIO(get("https://huggingface.co/datasets/Salesforce/wikitext/resolve/main/wikitext-2-raw-v1/train-00000-of-00001.parquet")))
paras = [t.strip() for t in df["text"] if len(t.strip()) > 300 and not t.strip().startswith("=")]; R.shuffle(paras)
for t in paras[:400]: add("wikitext", t[:1000])
R.shuffle(rows)
open("data/cal_general.jsonl", "w").write("".join(json.dumps(r) + "\n" for r in rows))
import collections; print(len(rows), collections.Counter(r["src"] for r in rows))
