"""how 'fitted-lookup-like' is each carrier? same 25 annotations (draw 0, source A), per carrier: layers touched, fraction of all model weights changed, relative Frobenius norm of the delta inside the touched down_proj blocks, relative norm of delta vs the whole layer-stack, max |delta|/|w| per weight, stored bytes."""
import sys, json, torch, hashlib, random
sys.path.insert(0, "/workspace/tp/v2/bonsai/ship")
import bonsai_loader as BL, common, feedback as FB
from transformers import AutoTokenizer
art = "/workspace/art"; tok = AutoTokenizer.from_pretrained(art); m, _ = BL.load(art, patch=None)
TPL = json.load(open("/workspace/tp/v2/task_search/EASY/PLAN.json"))["SKILL7"]["prompt"]; T = {"prompt": TPL.replace("{X}", "{t}")}
H = lambda s: int(hashlib.sha256(s.encode()).hexdigest()[:8], 16)
ent = sorted([json.loads(l) for l in open("/workspace/extra/skill_new_a.jsonl")], key=lambda r: H("fbx-ent|" + r["id"]))[:60]
A = [dict(text=r["user"].split("Request: ", 1)[1], label="Recipe") for r in random.Random(1000 * 25).sample(ent, 25)]
cal = [json.loads(l)["query"] for l in open("/workspace/extra/skill_calib.jsonl")]; total = sum(p.numel() for p in m.parameters())
W0 = {li: m.model.layers[li].mlp.down_proj.weight.data.clone() for li in range(36)}
out = {}
for tag, cfg in [("cols_default", {}), ("lr_default", dict(carrier="lr")), ("scale_lam100", dict(carrier="scale", lam=100.0)), ("cols_L18_c256", dict(layers=18, cols=256)), ("cols_L6_c128", dict(cols=128))]:
    P = FB.compute_patch(m, tok, A, T, cal, **cfg); ch = 0; num = den = 0.0; mx = 0.0; nl = 0
    for li in P["layers"]:
        w = m.model.layers[li].mlp.down_proj.weight.data; d = (w.float() - W0[li].float()); nz = d != 0; ch += int(nz.sum()); num += float(d.pow(2).sum()); den += float(W0[li].float().pow(2).sum()); mx = max(mx, float((d.abs() / W0[li].float().abs().clamp(min=1e-6))[nz].max())); nl += 1
    out[tag] = dict(layers_touched=nl, weights_changed=ch, frac_of_all_params=round(ch / total, 5), rel_fro_in_touched_blocks=round((num / den) ** .5, 4), max_rel_change_per_weight=round(mx, 3), bytes=FB.patch_bytes(P))
    print(tag, out[tag], flush=True)
    for li in P["layers"]: m.model.layers[li].mlp.down_proj.weight.data.copy_(W0[li])
json.dump(out, open("/workspace/res/dnorm.json", "w"), indent=1)
