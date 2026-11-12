"""L15 harness (deletion carrier) on SKILL7 + new label Recipe, same splits, draws and controls as the L1 locked protocol (v2/lanes/L1/run_locked_scale.py).
  python run_lane.py dev    ART ROOT OUT.json CFGS.json [ndraw] [limit]   # selection: ann_pool / sel only (never teach or exam)
  python run_lane.py locked ART ROOT OUT.json CFG.json  [ndraw]           # fixed config: locked teach (164), neighbours, exam after feedback, controls
ROOT = repo root with the exam files (LF line endings, sha-verified). ART = Bonsai pack dir or an HF model (proxy smoke tests).
Reference for retention = the UNPATCHED artifact's greedy outputs (for Bonsai = the released model; L1 showed the pack is output-identical).
dev:    real = 25 ann_pool items labelled Recipe; dev-teach = the other ann_pool items; dev-neighbours = base-correct sel[:300]; shuffled = 25 base-correct sel[300:] items labelled Recipe.
locked: real / shuffled (25 base-correct sel items labelled Recipe, as L1) / wrong (the 25 real items with a wrong OLD label) per draw (seeds 2500+d, the L1 draws);
        empty = no deletions file (byte-identical tree, checked through the shipped CLI in the e2e step of modal_l15.py)."""
import sys, os, json, random, time, collections, tempfile
mode, art, root, out, cfgp = sys.argv[1:6]; NDRAW = int(sys.argv[6]) if len(sys.argv) > 6 else (2 if mode == "dev" else 3); LIMIT = int(sys.argv[7]) if len(sys.argv) > 7 else 0
H = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, H); sys.path.insert(0, os.path.join(H, "..", "L1")); sys.path.insert(0, root + "/v2/exam")
import loader as LD, torch, numpy as np
import deletion as DL, common
from transformers import AutoTokenizer
LD.verify(); T0 = time.time(); log = lambda *x: print(f"[{time.time()-T0:6.0f}s]", *x, flush=True)
NEW = LD.new_label("SKILL7"); OLD = LD.labels("SKILL7"); CFGS = json.load(open(cfgp)); CFGS = CFGS if isinstance(CFGS, list) else [CFGS]
ann_pool = LD.load("SKILL7", "ann_pool"); sel = LD.load("SKILL7", "sel")
m = DL.load_model(art); tok = AutoTokenizer.from_pretrained(art); dev = m.device
BS = 16 if dev.type == "cuda" else 8
gen = lambda rows: common.generate(m, tok, [r["user"] for r in rows], bs=BS)
calib = [json.loads(l)["query"] for l in open(os.path.join(H, "..", "L1", "feedback_calib.jsonl"))]
R = dict(mode=mode, art=os.path.basename(art.rstrip("/")), cfgs=CFGS, ndraw=NDRAW, n_params=sum(p.numel() for p in m.parameters()))
cal = DL.calib_seqs(m, tok, calib); log("calib seqs", len(cal))
if LIMIT: ann_pool = ann_pool[:max(LIMIT, 40)]; sel = sel[:2 * LIMIT]
pool_out = dict(zip([r["id"] for r in ann_pool + sel[:600]], gen(ann_pool + sel[:600]))); log("base outputs on pool/sel done")
R["ann_pool_base_said"] = dict(collections.Counter(pool_out[r["id"]] for r in ann_pool).most_common(6))
sel_ok = [r for r in sel[:600] if pool_out[r["id"]] == r["kind"]]
def learn(rows, labels, cfg, d):
    A = [dict(query=r["user"], label=l) for r, l in zip(rows, labels)]; c = dict(cfg); c["seed"] = d
    D = DL.learn_deletions(m, tok, A, None, None, c, cal=cal, log=lambda h: log("  ", h))
    f = tempfile.NamedTemporaryFile(delete=False, suffix=".bin"); f.close(); DL.save(D, f.name); D["bytes"] = os.path.getsize(f.name)
    D2 = DL.load(f.name); os.remove(f.name); assert all(torch.equal(a, b) for a, b in zip(D["idx"], D2["idx"]))
    return D
def rate(rows, outs, lab=None): return float(np.mean([o == (lab or r["kind"]) for r, o in zip(rows, outs)])) if rows else None
def summ(D): s = D["stats"]; return {k: s[k] for k in ("n_deleted", "frac_of_target", "frac_of_all_params", "n_prior", "deleted_not_in_prior", "prior_restored", "skipped_steps", "seconds")} | {"bytes": D["bytes"], "hist_last": s["hist"][-1]}
R["runs"] = []
if mode == "dev":
    half = min(300, len(sel) // 2); nb = [r for r in sel[:half] if pool_out[r["id"]] == r["kind"]]; R["n_dev_neigh"] = len(nb); nbids = {r["id"] for r in sel[:half]}
    for ci, cfg in enumerate(CFGS):
        for d in range(NDRAW):
            rr = random.Random(9100 + d); ix = rr.sample(ann_pool, 25); ids = {r["id"] for r in ix}; dt = [r for r in ann_pool if r["id"] not in ids]
            ps = [r for r in sel_ok if r["id"] not in nbids]; shuf = rr.sample(ps, min(25, len(ps)))
            for arm, rows in (("real", ix), ("shuffled", shuf)):
                D = learn(rows, [NEW] * len(rows), cfg, d); sv = DL.apply(m, D)
                o_t = gen(dt); o_n = gen(nb); DL.undo(m, sv)
                rec = dict(cfg=ci, d=d, arm=arm, dev_teach=rate(dt, o_t, NEW), dev_teach_n=len(dt), dev_neigh_kept=rate(nb, o_n), dev_teach_out=collections.Counter(o_t).most_common(5)) | summ(D)
                R["runs"].append(rec); json.dump(R, open(out, "w")); log(ci, d, arm, {k: rec[k] for k in ("dev_teach", "dev_neigh_kept", "n_deleted")})
else:
    cfg = CFGS[0]; exam = LD.load("SKILL7", "exam"); teach = LD.load("SKILL7", "teach"); neigh_ids = set(LD.load("SKILL7", "neighbours"))
    if LIMIT: exam = exam[:LIMIT]; teach = teach[:LIMIT]
    ex_out = gen(exam); base_ok = {r["id"]: o == r["kind"] for r, o in zip(exam, ex_out)}; R["base_exam_acc"] = float(np.mean(list(base_ok.values())))
    ex_by_id = {r["id"]: r for r in exam}; neigh = [ex_by_id[i] for i in sorted(neigh_ids) if base_ok.get(i)]; R["n_neigh_base_correct"] = len(neigh)
    e = gen(teach); R["empty"] = dict(teach=rate(teach, e, NEW), teach_out=collections.Counter(e).most_common(5), neigh_kept=1.0, exam_retention=1.0, note="no deletions file = unpatched model (by construction)")
    log("base exam acc", R["base_exam_acc"], "empty teach", R["empty"]["teach"]); json.dump(R, open(out, "w"))
    for d in range(NDRAW):
        rr = random.Random(2500 + d); ix = rr.sample(ann_pool, 25); wl = [rr.choice([l for l in OLD if l != pool_out[r["id"]]]) for r in ix]; shuf = rr.sample([r for r in sel_ok if r in sel[:300]], 25)   # = L1 draws (sel_ok from sel[:300])
        for arm, rows, lab in (("real", ix, [NEW] * 25), ("shuffled", shuf, [NEW] * 25), ("wrong", ix, wl)):
            D = learn(rows, lab, cfg, d); sv = DL.apply(m, D); t = gen(teach); n = gen(neigh)
            rec = dict(d=d, arm=arm, teach=rate(teach, t, NEW), teach_out=collections.Counter(t).most_common(5), neigh_kept=rate(neigh, n), neigh_n=len(neigh),
                       neigh_by_kind={k: round(rate([r for r in neigh if r["kind"] == k], [x for r, x in zip(neigh, n) if r["kind"] == k]), 3) for k in sorted({r["kind"] for r in neigh})}) | summ(D)
            if arm == "wrong": rec["wrong_label_hit"] = float(np.mean([x == l for x, l in zip(gen(ix), wl)]))
            if (arm == "real" and d < 2) or (arm != "real" and d == 0):
                xo = gen(exam); xok = {r["id"]: o == r["kind"] for r, o in zip(exam, xo)}; rep = LD.report(exam, base_ok, xok)
                rec["exam_after"] = dict(overall=rep["overall"], by_kind={k: round(v["ret"], 4) for k, v in rep["by_kind"].items()}, gate_overall_98=rep["gate_overall_98"], gate_kind_95=rep["gate_kind_95"])
                rec["exam_identical_outputs"] = float(np.mean([a == b for a, b in zip(xo, ex_out)])); rec["recipe_on_exam"] = sum(o == NEW for o in xo); rec["off_menu_exam"] = sum(o not in OLD for o in xo)
            DL.undo(m, sv); R["runs"].append(rec); json.dump(R, open(out, "w")); log(d, arm, {k: rec.get(k) for k in ("teach", "neigh_kept", "n_deleted")})
R["done"] = True; R["seconds"] = round(time.time() - T0); json.dump(R, open(out, "w")); log("DONE")
