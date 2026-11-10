"""L1 locked-exam run, SCALE carrier (shipped default; config fixed in advance = feedback.py defaults: carrier scale, layers 6, lam 100, awt 10, margin 16, ridge 0.05; no tuning on locked data). Derived from run_locked.py (cols). Original docstring: L1 locked-exam run (SKILL7 shared exam). Order is fixed: (1) base HF fp16 Bonsai x3 (bs 8/4/16) on the exam, (2) the L1 artifact (shipped loader, no patch) ONCE on the exam,
(3) feedback protocol on the locked teach set (<=25 annotations from ann_pool; real/shuffled/wrong x10 draws, empty), neighbours from the loader, exam-after for draw 0.
python run_locked.py ART SNAPSHOT TPROOT OUT.json"""
import sys, os, json, random, time, gc, collections
art, snap, root, out = sys.argv[1:5]
sys.path.insert(0, root + "/v2/exam"); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "")
import loader as LD
import torch, numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
import bonsai_loader as BL, common, feedback as FB
LD.verify(); T0 = time.time(); log = lambda *x: print(f"[{time.time()-T0:6.0f}s]", *x, flush=True)
exam = LD.load("SKILL7", "exam"); teach = LD.load("SKILL7", "teach"); ann_pool = LD.load("SKILL7", "ann_pool"); neigh_ids = set(LD.load("SKILL7", "neighbours")); sel = LD.load("SKILL7", "sel")
NEW = LD.new_label("SKILL7"); OLD = LD.labels("SKILL7"); R = dict(n_exam=len(exam), n_teach=len(teach), n_ann_pool=len(ann_pool), n_neigh=len(neigh_ids))
tok = AutoTokenizer.from_pretrained(art)
def gen_with(m, rows, bs=16): return common.generate(m, tok, [r["user"] for r in rows], bs=bs)
ok_of = lambda rows, outs: {r["id"]: o == r["kind"] for r, o in zip(rows, outs)}
# (1) base: reference HF fp16 model, three batch sizes
base_runs = []; mb = AutoModelForCausalLM.from_pretrained(snap, torch_dtype=torch.float16).cuda().eval()
for bs in (8, 4, 16):
    o = gen_with(mb, exam, bs); base_runs.append(ok_of(exam, o)); log("base run bs", bs, "acc", np.mean(list(base_runs[-1].values())))
    if bs == 16: R["base_outputs_bs16"] = o
del mb; gc.collect(); torch.cuda.empty_cache()
agree = lambda a, b: float(np.mean([a[k] == b[k] for k in a])); R["base_run_acc"] = [float(np.mean(list(b.values()))) for b in base_runs]; R["base_pairwise_agree"] = [agree(base_runs[0], base_runs[1]), agree(base_runs[0], base_runs[2]), agree(base_runs[1], base_runs[2])]
base_ok = {k: all(b[k] for b in base_runs) for k in base_runs[0]}; base_maj = {k: sum(b[k] for b in base_runs) >= 2 for k in base_runs[0]}
# (2) candidate (L1 artifact, shipped answer path = shipped loader + same generate), ONCE
m, _ = BL.load(art, patch=None); cand_out = gen_with(m, exam); cand_ok = ok_of(exam, cand_out); log("candidate acc", np.mean(list(cand_ok.values())))
R["cand_acc"] = float(np.mean(list(cand_ok.values()))); R["cand_vs_base_bs16_identical_outputs"] = float(np.mean([a == b for a, b in zip(cand_out, R["base_outputs_bs16"])]))
for nm, bo in (("intersection_of_3_runs", base_ok), ("majority_of_3", base_maj)):
    R["report_" + nm] = LD.report(exam, bo, cand_ok); R["report_" + nm + "_clean_only"] = LD.report(LD.load("SKILL7", "exam", clean_only=True), bo, cand_ok)
json.dump(R, open(out, "w")); log("stage 1 saved")
# (3) feedback protocol
ex_by_id = {r["id"]: r for r in exam}; BASE_PRED = dict(zip([r["id"] for r in exam], R["base_outputs_bs16"]))
neigh = [ex_by_id[i] for i in neigh_ids if base_ok.get(i)]; R["n_neigh_base_correct"] = len(neigh)
calib = [json.loads(l)["query"] for l in open(art + "/feedback_calib.jsonl")]; cal = FB.calib_seqs(m, tok, calib)
pool_base = gen_with(m, ann_pool + sel[:300]); pb = dict(zip([r["id"] for r in ann_pool + sel[:300]], pool_base))
sel_ok = [r for r in sel[:300] if pb[r["id"]] == r["kind"]]; R["ann_pool_base_said"] = dict(collections.Counter(pb[r["id"]] for r in ann_pool).most_common(6))
CFG = dict(layers=FB.DEF["layers"], lam=FB.LAM_SCALE, awt=FB.DEF["awt"], margin=FB.DEF["margin"], ridge=FB.DEF["ridge"], carrier="scale"); R["cfg"] = CFG
def arm(rows, labels):
    A = [dict(query=r["user"], label=l) for r, l in zip(rows, labels)]; return FB.compute_patch(m, tok, A, None, calib, cal=cal, **CFG)
def ev(name, rows, lab=None): o = gen_with(m, rows); return dict(ok=[int(x == (lab or r["kind"])) for r, x in zip(rows, o)], out=o)
def bysrc(rows, ok):
    d = collections.defaultdict(list)
    for r, x in zip(rows, ok): d[r["src"]].append(x)
    return {k: [round(float(np.mean(v)), 3), len(v)] for k, v in d.items()}
NDRAW = 8; R["draws"] = []; R["empty"] = dict(teach=float(np.mean([BASE_PRED.get(r["id"]) == NEW for r in teach])) if False else None)
e = ev("teach_empty", teach, NEW); R["empty"] = dict(teach_acc=float(np.mean(e["ok"])), neigh_kept=1.0, note="no patch -> byte-identical tree (tested in e2e); teach evaluated on the unpatched candidate")
for d in range(NDRAW):
    rr = random.Random(2500 + d); ix = rr.sample(ann_pool, 25); wl = [rr.choice([l for l in OLD if l != pb[r["id"]]]) for r in ix]; shuf = rr.sample(sel_ok, 25)
    for name, rows, lab in (("real", ix, [NEW] * 25), ("shuffled", shuf, [NEW] * 25), ("wrong", ix, wl)):
        P = arm(rows, lab); t = ev("t", teach, NEW); rec = dict(d=d, arm=name, bytes=FB.patch_bytes(P), teach=float(np.mean(t["ok"])), teach_by_src=bysrc(teach, t["ok"]))
        if name in ("real", "wrong"):
            n = ev("n", neigh); rec.update(neigh_kept=float(np.mean(n["ok"])), neigh_n=len(neigh), neigh_by_kind={k: round(float(np.mean([x for r, x in zip(neigh, n["ok"]) if r["kind"] == k])), 3) for k in {r["kind"] for r in neigh}})
        rec["teach_new_label_count"] = sum(o == NEW for o in t["out"])
        if (name == "real" and d < 3) or (name in ("shuffled", "wrong") and d == 0):
            xo = gen_with(m, exam); xok = ok_of(exam, xo); rec["exam_after"] = LD.report(exam, base_ok, xok)
            rec["recipe_leak_exam"] = sum(o == NEW for o in xo); rec["off_menu_exam"] = sum(o not in OLD for o in xo); rec["exam_acc_after"] = float(np.mean(list(xok.values())))
            if name == "real" and d == 0: R["exam_after_feedback_draw0"] = rec["exam_after"]; R["teach_outputs_draw0"] = collections.Counter(t["out"]).most_common(8)
        if P: FB.undo(m, P)
        R["draws"].append(rec); json.dump(R, open(out, "w")); log(rec["d"], rec["arm"], {k: v for k, v in rec.items() if k in ("teach", "neigh_kept")})
R["done"] = True; json.dump(R, open(out, "w")); log("DONE")
