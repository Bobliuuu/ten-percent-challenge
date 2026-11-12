"""ONE locked run for deletion variant B (config chosen on DEV and passed in CFG json; k chosen per draw by leave-one-out over the 25 annotations).
python run_locked_del.py ART ROOT OUT.json CFG.json DRAWS(comma)"""
import sys, os, json, random, time, collections, numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, HERE + "/..")
art, root, out, cfgf, draws = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], [int(x) for x in sys.argv[5].split(",")]
sys.path.insert(0, root + "/v2/exam"); import loader as LD
import delcore as DC, bonsai_loader as BL, common, feedback as FB
from transformers import AutoTokenizer
CFG = json.load(open(cfgf)); T0 = time.time(); log = lambda *x: print(f"[{time.time()-T0:6.0f}s]", *x, flush=True)
LD.verify(); exam = LD.load("SKILL7", "exam"); teach = LD.load("SKILL7", "teach"); ann_pool = LD.load("SKILL7", "ann_pool"); neigh_ids = set(LD.load("SKILL7", "neighbours")); sel = LD.load("SKILL7", "sel")
NEW = LD.new_label("SKILL7"); OLD = LD.labels("SKILL7"); ex_by_id = {r["id"]: r for r in exam}
BASE = json.load(open(HERE + "/base_locked.json")); assert len(BASE["base_outputs_bs16"]) == len(exam)
base_ok = {r["id"]: o == r["kind"] for r, o in zip(exam, BASE["base_outputs_bs16"])}
tok = AutoTokenizer.from_pretrained(art); m, _ = BL.load(art, patch=None); NLAY = len(m.model.layers); LAYERS = list(range(NLAY - CFG["nl"], NLAY)); tR = tok(NEW, add_special_tokens=False).input_ids[0]
gen = lambda rows, mx=6: common.generate(m, tok, [r["user"] for r in rows], bs=32, mx=mx)
neigh = [ex_by_id[i] for i in neigh_ids if base_ok.get(i)]
calib = [json.loads(l)["query"] for l in open(art + "/feedback_calib.jsonl")] if os.path.exists(art + "/feedback_calib.jsonl") else [json.loads(l)["query"] for l in open(root + "/v2/lanes/L1/feedback_calib.jsonl")]
cal = FB.calib_seqs(m, tok, calib); flip_ref = DC.first_tok(m, tok, calib); tOLD = [tok(x, add_special_tokens=False).input_ids[0] for x in OLD]; CS = DC.calib_stats(m, tok, cal, LAYERS, [tR] + tOLD); DEL = DC.Dele(m, LAYERS); log("setup done", len(neigh), "neigh")
pool_base = gen(ann_pool + sel[:300]); pb = dict(zip([r["id"] for r in ann_pool + sel[:300]], pool_base)); sel_ok = [r for r in sel[:300] if pb[r["id"]] == r["kind"]]
R = dict(cfg=CFG, n_exam=len(exam), n_teach=len(teach), n_neigh=len(neigh), draws=[]); KS = CFG["KS"]
def bysrc(rows, ok):
    d = collections.defaultdict(list)
    for r, x in zip(rows, ok): d[r["src"]].append(x)
    return {k: [round(float(np.mean(v)), 3), len(v)] for k, v in d.items()}
for d in draws:
    rr = random.Random(2500 + d); ix = rr.sample(ann_pool, 25); wl = [rr.choice([l for l in OLD if l != pb[r["id"]]]) for r in ix]; shuf = rr.sample(sel_ok, 25)
    for name, rows, lab in (("real", ix, [NEW] * 25), ("shuffled", shuf, [NEW] * 25), ("wrong", ix, wl)):
        A = [dict(query=r["user"], label=l) for r, l in zip(rows, lab)]; GA = DC.ann_gain(m, tok, A, LAYERS)
        first = [tok(l, add_special_tokens=False).input_ids[0] for l in lab]; fn = DC.make_scores(CS, LAYERS, CFG['rule'], CFG['mu'], CFG['nu'], {t_: c_ / 25 for t_, c_ in collections.Counter(first).items()})
        k, tab, sc = DC.choose_k(m, tok, DEL, fn, GA, [r["user"] for r in rows], first, calib, flip_ref, KS, LAYERS)
        mask, kk = DC.topk_mask(sc, LAYERS, k); DEL.apply(mask); bm = DC.pack_bitmap(mask, LAYERS)
        t = gen(teach); tok_ok = [o == NEW for o in t]; rec = dict(d=d, arm=name, k=k, units=kk, bitmap_bytes=len(bm), k_table=tab, teach=float(np.mean(tok_ok)), teach_by_src=bysrc(teach, tok_ok), per_layer=[int(mask[li].sum()) if li in mask else 0 for li in LAYERS])
        if name in ("real", "wrong"):
            n = gen(neigh); nok = [o == r["kind"] for r, o in zip(neigh, n)]; rec.update(neigh_kept=float(np.mean(nok)), neigh_n=len(neigh), neigh_by_kind={kd: round(float(np.mean([x for r, x in zip(neigh, nok) if r["kind"] == kd])), 3) for kd in {r["kind"] for r in neigh}})
        if (name == "real" and d < 3) or (name in ("shuffled", "wrong") and d == 0):
            xo = gen(exam); xok = {r["id"]: o == r["kind"] for r, o in zip(exam, xo)}; rec["exam_after"] = LD.report(exam, base_ok, xok); rec["recipe_leak_exam"] = sum(o == NEW for o in xo); rec["off_menu_exam"] = sum(o not in OLD for o in xo); rec["exam_acc_after"] = float(np.mean(list(xok.values())))
        DEL.restore(); R["draws"].append(rec); json.dump(R, open(out, "w")); log({k_: v for k_, v in rec.items() if k_ in ("d", "arm", "k", "units", "teach", "neigh_kept", "exam_acc_after")})
R["done"] = True; json.dump(R, open(out, "w")); print("RESULT_JSON", json.dumps(R), flush=True)
