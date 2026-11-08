import sys, os, json, random, torch
sys.path.insert(0, "/workspace/tp/v2/bonsai/ship")
import bonsai_loader as BL, common, feedback as FB
from transformers import AutoTokenizer
art = "/workspace/art"; tok = AutoTokenizer.from_pretrained(art); T = json.load(open("/workspace/e2e/task.json"))
cal = [json.loads(l)["query"] for l in open(art + "/feedback_calib.jsonl")]
A = [json.loads(l) for l in open("/workspace/e2e/c0.jsonl")]; B = [json.loads(l) for l in open("/workspace/e2e/c1.jsonl")]
m, _ = BL.load(art, patch=None)
P1 = FB.compute_patch(m, tok, A, T, cal); P1.pop("orig"); P2 = FB.compute_patch(m, tok, B, T, cal); P2.pop("orig")
Wf = {li: m.model.layers[li].mlp.down_proj.weight.data.clone() for li in P1["layers"]}
M = FB.merge_patch(P1, P2); torch.save(M, "/workspace/dbg_merged.pt"); del m; torch.cuda.empty_cache()
m2, _ = BL.load(art, patch="/workspace/dbg_merged.pt")
for li in P1["layers"]:
    w2 = m2.model.layers[li].mlp.down_proj.weight.data; d = (w2.float() - Wf[li].float()).abs(); print(li, "max diff", float(d.max()), "n diff", int((d > 0).sum()))
