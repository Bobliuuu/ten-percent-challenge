"""usage: eval_prune.py MODEL SAL.pt ITEMS.jsonl OUTDIR cfg1,cfg2,...   cfg = base | METHOD:FRAC  (METHOD in freq,reap,ean,gate,rand) e.g. reap:0.5
Mask-prune (== physical removal), greedy generate with native template, score with tournament scorer. Saves per-item rows."""
import sys, json, time, torch, os
sys.path.insert(0, "/opt/dlami/nvme/work/l2/code"); sys.path.insert(0, "/opt/dlami/nvme/work/l2/data")
from moe_lib import *
from scorer import score_item
from transformers import AutoTokenizer, AutoModelForCausalLM
model_id, salf, itemsf, outd, cfgs = sys.argv[1:6]
bs = int(os.environ.get("BS", 16)); maxnew = int(os.environ.get("MAXNEW", 320)); kw = json.loads(os.environ.get("TKW", "{}"))
os.makedirs(outd, exist_ok=True)
tok = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto"); model.eval()
L = patch(model)
sal = torch.load(salf) if salf != "none" else None
items = [json.loads(l) for l in open(itemsf)]
def masks(method, frac, seed=0):
    m = {}
    g = torch.Generator().manual_seed(seed)
    for l in range(sal["cnt"].shape[0]):
        E = sal["cnt"].shape[1]; k = int(round(E * frac))
        sc = {"freq": sal["cnt"][l], "reap": sal["reap"][l], "ean": sal["ean"][l], "gate": sal["gatesum"][l], "rand": torch.rand(E, generator=g)}[method]
        if method in ("reap", "ean"): sc = torch.where(sal["cnt"][l] == 0, torch.full_like(sc, -1.0), sc)
        pr = torch.zeros(E, dtype=torch.bool); pr[sc.argsort()[:k]] = True; m[l] = pr
    return m
for cfg in cfgs.split(","):
    fn = f"{outd}/{cfg.replace(':','_')}.json"
    if os.path.exists(fn): continue
    STATE["mask"] = None if cfg == "base" else masks(*(lambda a: (a[0], float(a[1])))(cfg.split(":")))
    t0 = time.time()
    outs = generate(model, tok, items, bs=bs, max_new=maxnew, kw=kw, log=sys.stderr)
    rows = []
    for it, o in zip(items, outs):
        s = score_item(o, it); rows.append(dict(id=it["id"], source=it["source"], kind=it["kind"], correct=bool(s["correct"]), parse_fail=bool(s["parse_fail"]), out=o[:600]))
    acc = sum(r["correct"] for r in rows) / len(rows)
    json.dump(dict(cfg=cfg, model=model_id, secs=time.time() - t0, acc=acc, rows=rows), open(fn, "w"))
    print("RESULT", cfg, acc, time.time() - t0, flush=True)
