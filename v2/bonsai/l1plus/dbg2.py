import sys, json, torch
sys.path.insert(0, "/workspace/tp/v2/bonsai/ship")
import bonsai_loader as BL, common
from transformers import AutoTokenizer
art = "/workspace/art"; tok = AutoTokenizer.from_pretrained(art); m, _ = BL.load(art, patch=None)
items = [json.loads(l) for l in open("/workspace/tp/v2/task_search/EASY/items_dev.jsonl")]; items = [r for r in items if r["task"] == "SKILL7"]
o0 = common.generate(m, tok, [r["user"] for r in items]); ok = [r for r, o in zip(items, o0) if o == r["kind"]]; ne = ok[::3]; print("base ok", len(ok), len(ne), flush=True)
BL.apply_patch_dict(m, torch.load("/workspace/dbg_merged.pt")); o1 = common.generate(m, tok, [r["user"] for r in ne]); print("neigh after merged patch (2 rounds, 25+25 items):", sum(o == r["kind"] for o, r in zip(o1, ne)) / len(ne))
import collections
bad = [(r["kind"], o, r["user"].split("Request: ")[1][:60]) for o, r in zip(o1, ne) if o != r["kind"]]; print(collections.Counter((a, b) for a, b, _ in bad).most_common(8)); print(bad[:6])
print(open("/workspace/e2e/c1.jsonl").read()[:900])
