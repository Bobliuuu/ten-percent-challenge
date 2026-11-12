"""Deletion carrier runs. DEV sweep: config chosen on teach_sel/neigh_sel only. LOCKED: locked teach + neighbours + full exam retention, controls.
  python run_del.py --art ART --extra EXTRA --out OUT.json --mode sweep|locked --cfgs cfgs.json [--draws 0,1,2,3] [--root REPO]"""
import sys, os, json, hashlib, random, time, argparse, collections
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, HERE + "/..")
import torch, numpy as np
import bonsai_loader as BL, common, del_lib as DL
from transformers import AutoTokenizer
ap = argparse.ArgumentParser(); ap.add_argument("--art", required=True); ap.add_argument("--extra", required=True); ap.add_argument("--out", required=True); ap.add_argument("--mode", default="sweep")
ap.add_argument("--cfgs", required=True); ap.add_argument("--draws", default="0,1"); ap.add_argument("--root", default=os.path.abspath(HERE + "/../../../.."))
a = ap.parse_args(); T0 = time.time(); log = lambda *x: print(f"[{time.time()-T0:6.0f}s]", *x, flush=True)
H = lambda s: int(hashlib.sha256(s.encode()).hexdigest()[:8], 16); rd = lambda p: [json.loads(l) for l in open(p) if l.strip()]
TPL = json.load(open(a.root + "/v2/task_search/EASY/PLAN.json"))["SKILL7"]["prompt"]; TASK = {"prompt": TPL.replace("{X}", "{t}")}
def norm(r, src=None): return dict(id=r["id"], text=r["user"], kind=r["kind"], src=r.get("src") or src)
items = [norm(r) for r in rd(a.root + "/v2/task_search/EASY/items_dev.jsonl") if r["task"] == "SKILL7"]
ent = sorted([norm(r) for r in rd(a.extra + "/skill_new_a.jsonl")], key=lambda r: H("fbx-ent|" + r["id"])); bbc = sorted([norm(r) for r in rd(a.extra + "/skill_new_b.jsonl")], key=lambda r: H("fbx-bbc|" + r["id"]))
old = [norm(r) for r in rd(a.extra + "/skill_old.jsonl")]; OLD = ["Alarm", "Weather", "Music", "Calendar", "Lists", "News", "Transport"]; NEW = "Recipe"
E_ANN, E_TSEL, E_TREP = ent[:60], ent[60:100], ent[100:]; B_TEACH = bbc[40:]
calib = [json.loads(l)["query"] for l in open(a.extra + "/skill_calib.jsonl")]
m, _ = BL.load(a.art, patch=None); tok = AutoTokenizer.from_pretrained(a.art); log("loaded")
def gen(rows, key="text"): return common.generate(m, tok, [r[key] for r in rows])
ok = lambda rows, outs, lab=None: [int(o == (lab or r["kind"])) for r, o in zip(rows, outs)]
ref = DL.first_logp(m, tok, calib, common.chat); log("ref logp", tuple(ref.shape))
RES = dict(mode=a.mode, runs=[]); save = lambda: json.dump(RES, open(a.out, "w"))
def tx(r): return r["text"].split("Request: ", 1)[1] if "Request: " in r["text"] else r["text"]
def run_cfg(cfg, rows, tag):
    """train a deletion mask on rows labelled NEW, return (dm, mods, hist, secs, bytes). Leaves the mask baked in the model; caller must unbake."""
    layers = list(range(m.config.num_hidden_layers - cfg["layers"], m.config.num_hidden_layers)); mods = DL.attach(m, layers, cfg["mods"]); t = time.time()
    prompts = [TASK["prompt"].replace("{t}", tx(r)) for r in rows]
    dm, hist = DL.train_mask(m, tok, common.chat, prompts, NEW, calib, ref, mods, cfg, log=log, seed=cfg.get("seed", 0))
    ids = DL.ids_of(dm); shapes = {f"{k[0]}.{k[1]}": DL.n_groups(d) for k, d in mods.items()}; buf = DL.encode(ids, shapes); assert DL.decode(buf) == ids
    nd = sum(len(v) for v in ids.values()); tot = sum(shapes.values()); DL.bake(mods, dm)
    return dm, mods, hist, time.time() - t, dict(bytes=len(buf), n_deleted=nd, n_groups=tot, frac=nd / tot, weights_zeroed=nd * 128), ids
if a.mode == "sweep":
    cfgs = json.load(open(a.cfgs)); draws = [int(x) for x in a.draws.split(",")]
    BASE = dict(zip([r["id"] for r in items], gen(items))); NB = [r for r in items if BASE[r["id"]] == r["kind"]]; N_SEL = [r for r in NB if H("fbx-nb|" + r["id"]) % 2 == 0]; log("neigh sel", len(N_SEL))
    SHUF = [r for r in old if gen([r])[0] == r["kind"]] if False else old
    for cfg in cfgs:
        for d in draws:
            for arm in ("real", "shuffled"):
                if arm == "shuffled" and not cfg.get("shuf", True): continue
                rows = random.Random(1000 * 25 + d).sample(E_ANN, 25) if arm == "real" else random.Random(7000 + 25000 + d).sample(old, 25)
                cfg2 = dict(cfg, seed=d); dm, mods, hist, secs, info, _ = run_cfg(cfg2, rows, cfg["tag"])
                t = gen(E_TSEL); res = dict(teach_sel=float(np.mean(ok(E_TSEL, t, NEW))), pred=collections.Counter(t).most_common(4))
                if arm == "real": n = gen(N_SEL); res["neigh_sel"] = float(np.mean(ok(N_SEL, n)))
                DL.unbake(mods); r = dict(tag=cfg["tag"], arm=arm, d=d, cfg=cfg, secs=round(secs), final=hist[-1], **info, **res); RES["runs"].append(r); save(); log("RESULT", json.dumps({k: v for k, v in r.items() if k not in ("cfg",)}))
elif a.mode == "locked":
    sys.path.insert(0, a.root + "/v2/exam"); import loader as LD; LD.verify()
    cfg = json.load(open(a.cfgs))[0]; draws = [int(x) for x in a.draws.split(",")]
    teach = LD.load("SKILL7", "teach"); pool = LD.load("SKILL7", "ann_pool"); nid = set(LD.load("SKILL7", "neighbours")); exam = LD.load("SKILL7", "exam"); ex = {r["id"]: r for r in exam}
    sel = LD.load("SKILL7", "sel")[:300]
    gx = lambda rows: gen(rows, "user")
    ex_out = gx(exam); base_ok = {r["id"]: o == r["kind"] for r, o in zip(exam, ex_out)}; BASE_OUT = {r["id"]: o for r, o in zip(exam, ex_out)}
    RES["base_exam_acc"] = float(np.mean(list(base_ok.values()))); log("base exam acc", RES["base_exam_acc"])
    neigh = [ex[i] for i in nid if base_ok[i]]; sel_ok = [r for r in sel if base_ok.get(r["id"], False)] or [r for r in sel if gx([r])[0] == r["kind"]]
    RES["n_neigh_base_correct"] = len(neigh); tb = gx(teach); RES["base_teach"] = float(np.mean(ok(teach, tb, NEW)))
    txu = lambda r: r["user"].split("Request: ", 1)[1]
    for d in draws:
        for arm in ("real", "shuffled", "wrong"):
            rr = random.Random(2500 + d); ix = rr.sample(pool, 25); shuf = rr.sample(sel_ok, 25); rows = shuf if arm == "shuffled" else ix
            if arm == "wrong":
                BASEP = dict(zip([r["id"] for r in ix], gx(ix))); wl = [rr.choice([l for l in OLD if l != BASEP[r["id"]]]) for r in ix]
            layers = list(range(m.config.num_hidden_layers - cfg["layers"], m.config.num_hidden_layers)); mods = DL.attach(m, layers, cfg["mods"]); t0 = time.time()
            prompts = [TASK["prompt"].replace("{t}", txu(r)) for r in rows]; label = wl if arm == "wrong" else NEW
            dm, hist = DL.train_mask(m, tok, common.chat, prompts, label, calib, ref, mods, dict(cfg, seed=d), log=log, seed=d)
            ids = DL.ids_of(dm); shapes = {f"{k[0]}.{k[1]}": DL.n_groups(x) for k, x in mods.items()}; buf = DL.encode(ids, shapes); assert DL.decode(buf) == ids
            nd = sum(len(v) for v in ids.values()); tot = sum(shapes.values()); DL.bake(mods, dm)
            tout = gx(teach); rec = dict(d=d, arm=arm, bytes=len(buf), n_deleted=nd, frac=nd / tot, teach=float(np.mean(ok(teach, tout, NEW))), new_label_count=sum(o == NEW for o in tout), n_train=len(prompts), train_label=(label if isinstance(label, str) else "wrong-old"), secs=round(time.time() - t0))
            bs = collections.defaultdict(list)
            for r, x in zip(teach, ok(teach, tout, NEW)): bs[r["src"]].append(x)
            rec["teach_by_src"] = {k: [round(float(np.mean(v)), 3), len(v)] for k, v in bs.items()}
            if arm in ("real", "wrong"):
                no = gx(neigh); rec["neigh_kept"] = float(np.mean(ok(neigh, no))); rec["neigh_n"] = len(neigh)
            xo = gx(exam); xok = {r["id"]: o == r["kind"] for r, o in zip(exam, xo)}
            rec["exam_after"] = LD.report(exam, base_ok, xok); rec["exam_acc_after"] = float(np.mean(list(xok.values()))); rec["recipe_leak_exam"] = sum(o == NEW for o in xo); rec["off_menu_exam"] = sum(o not in OLD for o in xo)
            ch = collections.Counter((r["kind"], BASE_OUT[r["id"]], o) for r, o in zip(exam, xo) if o != BASE_OUT[r["id"]]); rec["changed"] = [[list(k), v] for k, v in ch.most_common(25)]
            rec["alarm_n_base_correct"] = sum(1 for r in exam if r["kind"] == "Alarm" and base_ok[r["id"]]); rec["alarm_flipped"] = sum(v for k, v in ch.items() if k[0] == "Alarm" and k[1] == "Alarm"); rec["alarm_to_timer"] = sum(v for k, v in ch.items() if k[0] == "Alarm" and k[2] == "Timer")
            DL.unbake(mods); RES["runs"].append(rec); save(); log("RESULT", json.dumps({k: v for k, v in rec.items() if k not in ("changed", "exam_after")}))
            if arm == "real": open(os.path.splitext(a.out)[0] + f".mask_d{d}.bin", "wb").write(buf)
RES["done"] = True; save(); log("DONE")
