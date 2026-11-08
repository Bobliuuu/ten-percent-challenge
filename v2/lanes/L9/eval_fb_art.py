"""Feedback controls on the PACKED artifact: python eval_fb_art.py ART DATA_DIR OUT_JSON ; conditions empty, real, shuffled x2, wrong; real carrier saved to OUT_JSON.carrier.safetensors."""
import sys, os, json, random, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch, pack235 as P, feedback as FB
from safetensors.torch import save_file
ART, DD, OUTJ = sys.argv[1:4]; BASE = os.environ.get("L9_BASEDIR", "/data/out/res")
L = lambda n: [json.loads(l) for l in open(f"{DD}/skill7_{n}.jsonl")]
ann, teach, cal = L("fb_annot"), L("fb_teach"), L("cal"); sel, rep = L("sel"), L("rep")
m, tok = P.build_model(ART, os.environ.get("L9_GPUMEM", "120GiB"))
def run(rows):
    order = sorted(range(len(rows)), key=lambda i: len(rows[i]["user"])); outs = [None] * len(rows)
    for s in range(0, len(order), 64):
        idx = order[s:s + 64]
        for i, o in zip(idx, P.answer_text(m, tok, [rows[i]["user"] for i in idx])): outs[i] = o
    return outs
nb = []
for sn, rows in (("SEL", sel), ("REP", rep)):
    b = json.load(open(f"{BASE}/base_skill7_{sn}_1.json")); nb += [r for r, o in zip(rows, b["outs"]) if o == r["answer"]]
def ev(tag):
    t = run(teach); n = run(nb); ok = [o == r["answer"] for o, r in zip(n, nb)]; cells = collections.defaultdict(lambda: [0, 0])
    for o, r in zip(n, nb): c = cells[r["src"] + "|" + r["kind"]]; c[0] += 1; c[1] += int(o == r["answer"])
    d = dict(tag=tag, teach=sum(o == "Home" for o in t) / len(t), neigh=sum(ok) / len(ok), n_neigh=len(ok), min_cell=min(v[1] / v[0] for v in cells.values()), leak=sum(o == "Home" for o in n) / len(n), teach_outs=dict(collections.Counter(t).most_common(5)))
    print(d, flush=True); return d
res = []
co = run(cal); ERR = [r for r, o in zip(cal, co) if o != r["answer"]]; print("old-skill errors on cal:", len(ERR), flush=True)
def annot(seed):
    rg = random.Random(100 + seed); return rg.sample(ann, 12) + rg.sample(ERR, min(13, len(ERR)))
def apply(car, sign=1):
    for k, v in car.items(): w = m.model.layers[int(k.split(".")[1])].self_attn.o_proj.weight; w.data.add_(sign * v.to(w.device, w.dtype))
BK = {l: m.model.layers[l].self_attn.o_proj.weight.data.clone() for l in FB.LAYERS}
res.append(ev("empty"))
for mode, seed in (("real", 0), ("real", 1), ("real", 2), ("shuffled", 0), ("shuffled", 1), ("shuffled", 2), ("wrong", 0), ("wrong", 1)):
    a = annot(seed)
    if mode == "wrong":
        sk = ["Alarm", "Weather", "Music", "Calendar", "Lists", "News", "Transport"]; rg = random.Random(seed); a = [dict(r, answer=rg.choice([x for x in sk + ["Home"] if x != r["answer"]])) for r in a]; mode_ = "real"
    else: mode_ = mode
    car = FB.make_carrier(m, tok, a, cal, mode_, seed)   # NOTE make_carrier already adds the edit in place
    if mode == "real" and seed == 0: save_file(car, OUTJ + ".carrier.safetensors")
    d = ev(f"{mode}{seed}"); d["n_ann"] = len(a); d["carrier_bytes"] = sum(v.numel() * 2 for v in car.values()); res.append(d)
    for l in FB.LAYERS: m.model.layers[l].self_attn.o_proj.weight.data.copy_(BK[l])   # exact undo
json.dump(res, open(OUTJ, "w"), indent=1)
