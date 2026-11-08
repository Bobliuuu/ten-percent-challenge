"""tok_diag.py : retokenisation impact of vocab truncation (parent tokenizer vs trimmed) per task/source on all_items.jsonl"""
import json, collections, sys
from tokenizers import Tokenizer
A = Tokenizer.from_file("/home/ubuntu/work/v_k16_n140k/../hf/hub/models--Qwen--Qwen3.8-27B/snapshots/" + __import__("os").listdir("/home/ubuntu/work/hf/hub/models--Qwen--Qwen3.8-27B/snapshots")[0] + "/tokenizer.json")
B = Tokenizer.from_file("/home/ubuntu/work/v_k16_n140k/tokenizer.json")
st = collections.defaultdict(lambda: [0, 0, 0, 0])
for l in open("/home/ubuntu/work/all_items.jsonl"):
    r = json.loads(l); a = A.encode(r["user"]).ids; b = B.encode(r["user"]).ids; s = st[(r["task"], r["source"])]
    s[0] += 1; s[1] += len(a); s[2] += len(b); s[3] += any(x >= 140000 for x in a)
out = {f"{k[0]}|{k[1]}": dict(n=v[0], tok_base=v[1] / v[0], tok_trim=v[2] / v[0], inflation=v[2] / v[1], uses_removed_frac=v[3] / v[0]) for k, v in sorted(st.items())}
json.dump(out, open("/home/ubuntu/work/tok_diag.json", "w"), indent=1); print(json.dumps(out)[:3000])
