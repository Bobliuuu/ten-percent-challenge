"""DEV sweep for deletion variant B (no locked file touched). One process = (seed, candidate layers N) x configs x k grid.
python dev_sweep.py ART OUT.json SEED NLAYERS [smoke]   (run through modal_dev.py)"""
import sys, os, json, random, time, collections, numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, HERE + "/..")
import delcore as DC, bonsai_loader as BL, common, feedback as FB
from transformers import AutoTokenizer
art, out, seed, NL = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]); smoke = len(sys.argv) > 5 and sys.argv[5] == 'smoke'
ROOT = os.path.abspath(HERE + "/../../../.."); T0 = time.time(); log = lambda *x: print(f"[{time.time()-T0:6.0f}s]", *x, flush=True)
rd = lambda p: [json.loads(l) for l in open(p)]
FS = ROOT + "/v2/lanes/L10/data/fb_skill/"; NEW = "Recipe"; OLD = ["Alarm", "Weather", "Music", "Calendar", "Lists", "News", "Transport"]
pool = rd(FS + "skill_new_a.jsonl") + rd(FS + "skill_new_b.jsonl"); old = rd(FS + "skill_old.jsonl")
tok = AutoTokenizer.from_pretrained(art); m, _ = BL.load(art, patch=None); NLAY = len(m.model.layers); LAYERS = list(range(NLAY - NL, NLAY)); tR = tok(NEW, add_special_tokens=False).input_ids[0]
calib = [json.loads(l)["query"] for l in open(ROOT + "/v2/lanes/L1/feedback_calib.jsonl")]
gen = lambda rows, mx=6: common.generate(m, tok, [r["user"] for r in rows], bs=32, mx=mx)
rr = random.Random(1000 + seed)
# dev neighbours: base-correct old items; shuffled pool: other base-correct old items
o_sh = rr.sample(old, 700); ob = gen(o_sh); okold = [r for r, o in zip(o_sh, ob) if o == r["kind"]]; neigh = okold[:250]; shufpool = okold[250:]; log("base acc old", len(okold) / 700)
ann = rr.sample(pool, 25); rest = [r for r in pool if r["id"] not in {a["id"] for a in ann}]; test = rr.sample(rest, 150 if not smoke else 40)
mk = lambda rows, lab=None: [dict(query=r["user"], label=lab or r["kind"]) for r in rows]
shuf = rr.sample(shufpool, 25)
cal = FB.calib_seqs(m, tok, calib); log("cal", len(cal))
base_flip_ref = DC.first_tok(m, tok, calib)
CS = DC.calib_stats(m, tok, cal, LAYERS, tR); log("calib stats done")
GA = DC.ann_gain(m, tok, mk(ann, NEW), LAYERS); GS = DC.ann_gain(m, tok, mk(shuf, NEW), LAYERS); wl = [rr.choice([l for l in OLD if l != r['kind']]) for r in ann]; GW = DC.ann_gain(m, tok, [dict(query=r['user'], label=l) for r, l in zip(ann, wl)], LAYERS); log("annotation gains done", {li: None if GA[li] is None else tuple(GA[li].shape) for li in LAYERS[:1]})
DEL = DC.Dele(m, LAYERS); ann_prompts = [r["user"] for r in ann]
KS = [30, 100, 300, 1000, 2000, 4000, 10000, 30000] if not smoke else [3000, 30000]
CFG = [("gain", 0, 0), ("ratio", 0, 0), ("wandaratio", 0, 0)] + [("contrast", mu, nu) for mu in (0, 1, 3, 10, 30) for nu in (0, 1, 3)][1:] + [("wanda", 0, nu) for nu in (1, 3, 10)]
if os.environ.get('RULESET') == '2': CFG = [('tstat', mu, 0) for mu in (0.1, 0.3, 1)] + [('spec', mu, nu) for mu in (0.5, 1, 2, 4) for nu in (0, 1)]
if smoke: CFG = CFG[:2]
def sel_from(G_, excl=None):
    n = G_[LAYERS[0]].shape[0]; keep = [i for i in range(n) if i != excl]
    return {li: G_[li][keep].mean(0) for li in LAYERS}
acc = lambda rows, outs, lab=None: float(np.mean([o == (lab or r["kind"]) for r, o in zip(rows, outs)]))
R = dict(seed=seed, layers=NL, ann=[a["id"] for a in ann], rows=[], n_test=len(test), n_neigh=len(neigh))
for rule, mu, nu in CFG:
    fn = DC.make_scores(CS, LAYERS, rule, mu, nu); sc_full = fn(sel_from(GA)); sc_sh = fn(sel_from(GS)); sc_w = fn(sel_from(GW))
    loo_scores = [fn(sel_from(GA, j)) for j in range(25)] if len(KS) else []
    for k in KS:
        mask, kk = DC.topk_mask(sc_full, LAYERS, k)
        if kk == 0: R["rows"].append(dict(rule=rule, mu=mu, nu=nu, k=k, kk=0)); continue
        # leave-one-out over the annotations: held-out first-token hit
        hits = []
        for j in range(25):
            mj, _ = DC.topk_mask(loo_scores[j], LAYERS, k); DEL.apply(mj); hits.append(DC.first_tok(m, tok, [ann_prompts[j]])[0] == tR)
        DEL.apply(mask); flip = float(np.mean([a != b for a, b in zip(DC.first_tok(m, tok, calib), base_flip_ref)]))
        tt = gen(test); nn = gen(neigh); ta = [o == NEW for o in tt]; bm = DC.pack_bitmap(mask, LAYERS)
        msh, _ = DC.topk_mask(sc_sh, LAYERS, k); DEL.apply(msh); sh = gen(test[:100]); mw, _ = DC.topk_mask(sc_w, LAYERS, k); DEL.apply(mw); wt_ = gen(test[:100]); wn = gen(neigh); DEL.restore()
        rec = dict(rule=rule, mu=mu, nu=nu, k=k, kk=kk, loo_hit=float(np.mean(hits)), calib_flip=flip, teach=float(np.mean(ta)), teach_A=float(np.mean([t for t, r in zip(ta, test) if r["src"] == "massive"] or [float("nan")])),
                   teach_B=float(np.mean([t for t, r in zip(ta, test) if r["src"] != "massive"] or [float("nan")])), neigh=acc(neigh, nn), shuffled=acc(test[:100], sh, NEW), bitmap_bytes=len(bm), wrong_teach=acc(test[:100], wt_, NEW), wrong_neigh=acc(neigh, wn), neigh_by_kind={kd: round(float(np.mean([o == r['kind'] for r, o in zip(neigh, nn) if r['kind'] == kd])), 3) for kd in OLD},
                   per_layer=[int(mask[li].sum()) if li in mask else 0 for li in LAYERS])
        R["rows"].append(rec); json.dump(R, open(out, "w")); log({k_: (round(v, 3) if isinstance(v, float) else v) for k_, v in rec.items() if k_ != "per_layer"})
R["done"] = True; json.dump(R, open(out, "w")); print("RESULT_JSON", json.dumps(R), flush=True)
