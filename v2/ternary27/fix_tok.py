"""re-trim tokenizer of an existing variant dir: fix_tok.py DIR NV"""
import sys, json, os
from huggingface_hub import snapshot_download
d, NV = sys.argv[1], int(sys.argv[2])
psrc = snapshot_download("Qwen/Qwen3.8-27B", allow_patterns=["tokenizer.json"])
tj = json.load(open(psrc + "/tokenizer.json")); m = tj["model"]
m["vocab"] = {t: i for t, i in m["vocab"].items() if i < NV}; kv = m["vocab"]
m["merges"] = [mg for mg in m["merges"] if (lambda a, b: a in kv and b in kv and (a + b) in kv)(*(mg if isinstance(mg, list) else mg.split(" ")))]
cur = json.load(open(d + "/tokenizer.json")); tj["added_tokens"] = cur["added_tokens"]
json.dump(tj, open(d + "/tokenizer.json", "w"), ensure_ascii=False); print(len(kv), len(m["merges"]))
