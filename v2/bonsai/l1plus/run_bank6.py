"""L1+ second task: BANK6 (bank-support routing, 6 kinds) + new label `Security`, Bonsai-8B, shared exam loader.
  python run_bank6.py ART SNAPSHOT TPROOT OUT.json CFG.json dev|locked
dev: no sealed file except the annotation pool (tuned-on source Banking77 only): annotate 25 of pool[:100], teach_dev = pool[100:], neighbours = BANK6 `sel` base-correct. Never reads exam/teach.
locked: (1) base HF fp16 x3 (bs 8/4/16) on the exam, (2) candidate = shipped loader without patch ONCE on the exam, (3) feedback protocol on the locked teach set + locked neighbours (10 draws real/shuffled/wrong, empty). NO exam-after-feedback scoring."""
import sys, os, json, random, time, gc, collections
art, snap, root, out, cfgf, mode = sys.argv[1:7]
sys.path.insert(0, root + "/v2/exam"); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/../ship")
import loader as LD
import torch, numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
import bonsai_loader as BL, common, feedback as FB
LD.verify(); T0 = time.time(); log = lambda *x: print(f"[{time.time()-T0:6.0f}s]", *x, flush=True)
CFG = json.load(open(cfgf)); CFG = {k: v for k, v in CFG.items() if k != "tag"}
TPL = LD.prompt("BANK6"); TASK = {"prompt": TPL.replace("{X}", "{t}")}; NEW = LD.new_label("BANK6"); OLD = LD.labels("BANK6")
tx = lambda r: r["user"].split("Message: ", 1)[1]
tok = AutoTokenizer.from_pretrained(art); R = dict(cfg=CFG, mode=mode)
def gen_with(m, rows, bs=16): return common.generate(m, tok, [r["user"] for r in rows], bs=bs)
ok_of = lambda rows, outs: {r["id"]: o == r["kind"] for r, o in zip(rows, outs)}
cal_rows = LD.load("BANK6", "cal"); calib = [r["user"] for r in cal_rows]
def bysrc(rows, ok):
    d = collections.defaultdict(list)
    for r, x in zip(rows, ok): d[r["src"]].append(x)
    return {k: [round(float(np.mean(v)), 3), len(v)] for k, v in d.items()}
if mode == "dev":
    m, _ = BL.load(art, patch=None); pool = LD.load("BANK6", "ann_pool"); sel = LD.load("BANK6", "sel"); cal = FB.calib_seqs(m, tok, calib)
    so = gen_with(m, sel); selk = [r for r, o in zip(sel, so) if o == r["kind"]]; R["base_sel_acc"] = len(selk) / len(sel); R["n_sel"] = len(sel); R["n_sel_base_correct"] = len(selk)
    ann_pool, teach_dev = pool[:100], pool[100:]; R["teach_dev_base"] = float(np.mean([o == NEW for o in gen_with(m, teach_dev)])); R["runs"] = []; log("base", R["base_sel_acc"], R["teach_dev_base"])
    for d in range(5):
        for arm in ("real", "shuffled"):
            rr = random.Random(100 + d); rows = rr.sample(ann_pool, 25) if arm == "real" else rr.sample(selk, 25)
            P = FB.compute_patch(m, tok, [dict(text=tx(r), label=NEW) for r in rows], TASK, calib, cal=cal, **CFG)
            t = gen_with(m, teach_dev); rec = dict(d=d, arm=arm, teach=float(np.mean([o == NEW for o in t])), bytes=FB.patch_bytes(P), pred=collections.Counter(t).most_common(4))
            if arm == "real": n = gen_with(m, selk); rec["neigh"] = float(np.mean([o == r["kind"] for r, o in zip(selk, n)])); rec["neigh_by_kind"] = {k: round(float(np.mean([o == r["kind"] for r, o in zip(selk, n) if r["kind"] == k])), 3) for k in OLD}
            FB.undo(m, P); R["runs"].append(rec); json.dump(R, open(out, "w")); log(rec)
    R["done"] = True; json.dump(R, open(out, "w")); sys.exit()
if mode == "fb":   # locked TEACH + NEIGHBOUR only (no exam scoring); neighbours base-correct = unpatched shipped model on the neighbour items
    m, _ = BL.load(art, patch=None); teach = LD.load("BANK6", "teach"); ann_pool = LD.load("BANK6", "ann_pool"); sel = LD.load("BANK6", "sel")
    ex = {r["id"]: r for r in LD.load("BANK6", "exam")}; na = [ex[i] for i in LD.load("BANK6", "neighbours")]; neigh = [r for r, o in zip(na, gen_with(m, na)) if o == r["kind"]]; R["n_neigh_base_correct"] = len(neigh)
    cal = FB.calib_seqs(m, tok, calib); pb = dict(zip([r["id"] for r in ann_pool + sel[:300]], gen_with(m, ann_pool + sel[:300]))); sel_ok = [r for r in sel[:300] if pb[r["id"]] == r["kind"]]; R["draws"] = []
    for d in range(10):
        rr = random.Random(2500 + d); ix = rr.sample(ann_pool, 25); shuf = rr.sample(sel_ok, 25)
        for name, rows in (("real", ix), ("shuffled", shuf)):
            P = FB.compute_patch(m, tok, [dict(text=tx(r), label=NEW) for r in rows], TASK, calib, cal=cal, **CFG); t = gen_with(m, teach); tk = [int(o == NEW) for o in t]
            rec = dict(d=d, arm=name, bytes=FB.patch_bytes(P), teach=float(np.mean(tk)), teach_by_src=bysrc(teach, tk))
            if name == "real": n = gen_with(m, neigh); nk = [int(o == r["kind"]) for o, r in zip(n, neigh)]; rec.update(neigh_kept=float(np.mean(nk)), neigh_by_kind={k: round(float(np.mean([x for r, x in zip(neigh, nk) if r["kind"] == k])), 3) for k in {r["kind"] for r in neigh}})
            FB.undo(m, P); R["draws"].append(rec); json.dump(R, open(out, "w")); log(d, name, {k: v for k, v in rec.items() if k in ("teach", "neigh_kept")})
    R["done"] = True; json.dump(R, open(out, "w")); sys.exit()
exam = LD.load("BANK6", "exam"); teach = LD.load("BANK6", "teach"); ann_pool = LD.load("BANK6", "ann_pool"); neigh_ids = set(LD.load("BANK6", "neighbours")); sel = LD.load("BANK6", "sel")
R.update(n_exam=len(exam), n_teach=len(teach), n_ann_pool=len(ann_pool), n_neigh=len(neigh_ids))
base_runs = []; mb = AutoModelForCausalLM.from_pretrained(snap, torch_dtype=torch.float16).cuda().eval()
for bs in (8, 4, 16):
    o = gen_with(mb, exam, bs); base_runs.append(ok_of(exam, o)); log("base run bs", bs, "acc", np.mean(list(base_runs[-1].values())))
    if bs == 16: R["base_outputs_bs16"] = o
del mb; gc.collect(); torch.cuda.empty_cache()
agree = lambda a, b: float(np.mean([a[k] == b[k] for k in a])); R["base_run_acc"] = [float(np.mean(list(b.values()))) for b in base_runs]; R["base_pairwise_agree"] = [agree(base_runs[0], base_runs[1]), agree(base_runs[0], base_runs[2]), agree(base_runs[1], base_runs[2])]
base_ok = {k: all(b[k] for b in base_runs) for k in base_runs[0]}; base_maj = {k: sum(b[k] for b in base_runs) >= 2 for k in base_runs[0]}
m, _ = BL.load(art, patch=None); cand_out = gen_with(m, exam); cand_ok = ok_of(exam, cand_out); log("candidate acc", np.mean(list(cand_ok.values())))
R["cand_acc"] = float(np.mean(list(cand_ok.values()))); R["cand_vs_base_bs16_identical_outputs"] = float(np.mean([a == b for a, b in zip(cand_out, R["base_outputs_bs16"])]))
for nm, bo in (("intersection_of_3_runs", base_ok), ("majority_of_3", base_maj)):
    R["report_" + nm] = LD.report(exam, bo, cand_ok); R["report_" + nm + "_clean_only"] = LD.report(LD.load("BANK6", "exam", clean_only=True), bo, cand_ok)
json.dump(R, open(out, "w")); log("stage 1 saved")
ex_by_id = {r["id"]: r for r in exam}; neigh = [ex_by_id[i] for i in neigh_ids if base_ok.get(i)]; R["n_neigh_base_correct"] = len(neigh)
cal = FB.calib_seqs(m, tok, calib); pb = dict(zip([r["id"] for r in ann_pool + sel[:300]], gen_with(m, ann_pool + sel[:300]))); sel_ok = [r for r in sel[:300] if pb[r["id"]] == r["kind"]]
R["ann_pool_base_said"] = dict(collections.Counter(pb[r["id"]] for r in ann_pool).most_common(6)); R["draws"] = []
e = gen_with(m, teach); R["empty"] = dict(teach_acc=float(np.mean([o == NEW for o in e])), neigh_kept=1.0, note="no patch -> byte-identical tree")
for d in range(10):
    rr = random.Random(2500 + d); ix = rr.sample(ann_pool, 25); wl = [rr.choice([l for l in OLD if l != pb[r["id"]]]) for r in ix]; shuf = rr.sample(sel_ok, 25)
    for name, rows, lab in (("real", ix, [NEW] * 25), ("shuffled", shuf, [NEW] * 25), ("wrong", ix, wl)):
        P = FB.compute_patch(m, tok, [dict(text=tx(r), label=l) for r, l in zip(rows, lab)], TASK, calib, cal=cal, **CFG); t = gen_with(m, teach); tk = [int(o == NEW) for o in t]
        rec = dict(d=d, arm=name, bytes=FB.patch_bytes(P), teach=float(np.mean(tk)), teach_by_src=bysrc(teach, tk), pred=collections.Counter(t).most_common(4))
        if name in ("real", "wrong"):
            n = gen_with(m, neigh); nk = [int(o == r["kind"]) for r, o in zip(neigh, n)]; rec.update(neigh_kept=float(np.mean(nk)), neigh_n=len(neigh), neigh_by_kind={k: round(float(np.mean([x for r, x in zip(neigh, nk) if r["kind"] == k])), 3) for k in {r["kind"] for r in neigh}})
        if P: FB.undo(m, P)
        R["draws"].append(rec); json.dump(R, open(out, "w")); log(d, name, {k: v for k, v in rec.items() if k in ("teach", "neigh_kept")})
R["done"] = True; json.dump(R, open(out, "w")); log("DONE")
