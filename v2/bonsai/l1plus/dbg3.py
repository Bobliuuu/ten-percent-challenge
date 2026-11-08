"""mitigation test for the Email<->Lists interference seen in the e2e continual run (same two annotation rounds c0, c1)"""
import sys, json, torch, collections
sys.path.insert(0, "/workspace/tp/v2/bonsai/ship")
import bonsai_loader as BL, common, feedback as FB
from transformers import AutoTokenizer
art = "/workspace/art"; tok = AutoTokenizer.from_pretrained(art); m, _ = BL.load(art, patch=None); T = json.load(open("/workspace/e2e/task.json"))
cal = [json.loads(l)["query"] for l in open(art + "/feedback_calib.jsonl")]
A = [json.loads(l) for l in open("/workspace/e2e/c0.jsonl")]; B = [json.loads(l) for l in open("/workspace/e2e/c1.jsonl")]
items = [json.loads(l) for l in open("/workspace/tp/v2/task_search/EASY/items_dev.jsonl")]; items = [r for r in items if r["task"] == "SKILL7"]
o0 = common.generate(m, tok, [r["user"] for r in items]); ok = [r for r, o in zip(items, o0) if o == r["kind"]]; ne = ok[::3]
em = [json.loads(l) for l in open("/workspace/extra/skill_email.jsonl")][60:200]
for tag, cfg in [("default", {}), ("awt30", dict(awt=30.0)), ("awt100", dict(awt=100.0)), ("lam3000_awt30", dict(lam=3000.0, awt=30.0)), ("scale_lam100", dict(carrier="scale", lam=100.0)), ("scale_lam100_awt100", dict(carrier="scale", lam=100.0, awt=100.0))]:
    Ps = []
    for ann in (A, B): Ps.append(FB.compute_patch(m, tok, ann, T, cal, **cfg))
    o1 = common.generate(m, tok, [r["user"] for r in ne]); te = common.generate(m, tok, [r["user"] for r in em])
    print(tag, "neigh", round(sum(o == r["kind"] for o, r in zip(o1, ne)) / len(ne), 3), "email teach", round(sum(o == "Email" for o in te) / len(te), 3), flush=True)
    for P in reversed(Ps): FB.undo(m, P)
