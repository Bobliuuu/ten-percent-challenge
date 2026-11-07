"""usage: gen_pruned.py SNAP SAL.pt ITEMS OUT.json METHOD:FRAC EBITS  -- physically prune (per-layer REAP/freq keep set), quantise kept experts (RTN g128 int8-held), real greedy generation, score with scorer.
BS/MAXNEW env. 4 GPUs split by layers via dispatch hooks."""
import sys, json, os, time, torch
sys.path.insert(0, "/opt/dlami/nvme/work/l2/code"); sys.path.insert(0, "/opt/dlami/nvme/work/l2/data")
from moe_lib import *
from pruned_loader import *
from scorer import score_item
from transformers import AutoTokenizer, AutoConfig
snap, salf, itemsf, outf, spec, ebits = sys.argv[1:7]; ebits = int(ebits)
cfg = AutoConfig.from_pretrained(snap); L, E = cfg.num_hidden_layers, cfg.num_experts
ng = torch.cuda.device_count(); dm = lambda k: torch.device(f"cuda:{0 if k=='embed' else ng-1 if k=='head' else min(ng-1, int(k)*ng//L)}")
keep = None
if spec != "base":
    sal = torch.load(salf); method, frac = spec.split(":"); frac = float(frac); keep = {}
    for l in range(L):
        k = E - int(round(E * frac))
        sc = {"freq": sal["cnt"][l], "reap": sal["reap"][l], "ean": sal["ean"][l]}[method].clone()
        if method != "freq": sc[sal["cnt"][l] == 0] = -1
        keep[l] = sorted(sc.argsort(descending=True)[:k].tolist())
tok = AutoTokenizer.from_pretrained(snap)
ab = int(os.environ.get('ABITS', 0))
qfn = (lambda n, w: rtn_fake(w, ab) if 'self_attn' in n and w.shape[-1] % 128 == 0 else w) if ab else None
model = load_pruned(snap, keep, dm, qfn=qfn, ebits=ebits)
mb = int(os.environ.get('MBITS', 0))
if mb:   # embeddings, lm_head (full vocab) and router rows fake-quantised to mb bits (g128)
    with torch.no_grad():
        for p in (model.model.embed_tokens.weight, model.lm_head.weight): p.copy_(rtn_fake(p, mb))
        for lay in model.model.layers: lay.mlp.gate.weight.copy_(rtn_fake(lay.mlp.gate.weight, mb))
add_dispatch_hooks(model, dm)
# MoE forward patched: needs _li, and experts count = len(experts)
patch(model)
for l, lay in enumerate(model.model.layers): lay.mlp.num_experts = len(lay.mlp.experts)
items = [json.loads(l) for l in open(itemsf)]
bs = int(os.environ.get("BS", 48)); mx = int(os.environ.get("MAXNEW", 256))
t0 = time.time(); outs = generate(model, tok, items, bs=bs, max_new=mx, log=sys.stderr)
rows = []
for it, o in zip(items, outs):
    s = score_item(o, it); rows.append(dict(id=it["id"], source=it["source"], kind=it["kind"], correct=bool(s["correct"]), parse_fail=bool(s["parse_fail"]), out=o[:600]))
json.dump(dict(cfg=f"{spec}:{ebits}", model=snap, secs=time.time() - t0, acc=sum(r["correct"] for r in rows) / len(rows), rows=rows), open(outf, "w"))
print("RESULT", spec, ebits, sum(r["correct"] for r in rows) / len(rows), time.time() - t0)
