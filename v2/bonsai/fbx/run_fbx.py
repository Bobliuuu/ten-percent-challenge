"""DEV experiments for the Bonsai column-sparse down_proj feedback (real greedy generation everywhere).
  python run_fbx.py --art ARTDIR --extra EXTRADATA --out OUT.json --mode sweep|final|policy [--cfgs cfgs.json] [--ndraw 10]
Splits are by sha256 of the item id (deterministic). NOTHING here reads a locked or sealed file: only the DEV items listed in v2/fb_edit/data_news + the extra DEV pools of build_data.py."""
import sys, os, json, hashlib, random, time, argparse, collections
HERE = os.path.dirname(os.path.abspath(__file__)); SHIP = HERE + "/../ship"; sys.path.insert(0, SHIP)
import torch, numpy as np
import bonsai_loader as BL, common, feedback as FB
from transformers import AutoTokenizer
ap = argparse.ArgumentParser(); ap.add_argument("--art", required=True); ap.add_argument("--extra", required=True); ap.add_argument("--out", required=True); ap.add_argument("--mode", default="sweep")
ap.add_argument("--task", default="news"); ap.add_argument("--calib"); ap.add_argument("--cfgs"); ap.add_argument("--ndraw", type=int, default=10); ap.add_argument("--ndraw-aux", type=int, default=5); a = ap.parse_args()
T0 = time.time(); log = lambda *x: print(f"[{time.time()-T0:6.0f}s]", *x, flush=True)
DEV = HERE + "/../../fb_edit/data_news/"
H = lambda s: int(hashlib.sha256(s.encode()).hexdigest()[:8], 16)
rd = lambda p: [json.loads(l) for l in open(p) if l.strip()]
def norm(r, src=None): return dict(id=r["id"], text=r["user"], kind=r["kind"], src=r.get("src") or src)
if a.task == "news":
    items = [norm(r) for r in rd(DEV + "items_dev.jsonl")]
    ent = sorted([norm(r) for f in ("fb_annot_25", "fb_teach_unseen_200") for r in rd(DEV + f + ".jsonl")], key=lambda r: H("fbx-ent|" + r["id"]))
    bbc = sorted([norm(r) for r in rd(a.extra + "/bbc_ent.jsonl")], key=lambda r: H("fbx-bbc|" + r["id"])); old = [norm(r) for f in ("old_ag", "old_hp") for r in rd(a.extra + "/" + f + ".jsonl")]
    OLD = ["World", "Sports", "Business", "Technology"]; NEW = "Entertainment"; SRC2 = "huffpost"; CALF = a.art + "/feedback_calib.jsonl"
else:   # SKILL7: new skill Recipe; source A = MASSIVE cooking, source B = CLINC150 cooking intents
    items = [norm(r) for r in rd(HERE + "/../../task_search/EASY/items_dev.jsonl") if r["task"] == "SKILL7"]
    ent = sorted([norm(r) for r in rd(a.extra + "/skill_new_a.jsonl")], key=lambda r: H("fbx-ent|" + r["id"])); bbc = sorted([norm(r) for r in rd(a.extra + "/skill_new_b.jsonl")], key=lambda r: H("fbx-bbc|" + r["id"]))
    old = [norm(r) for r in rd(a.extra + "/skill_old.jsonl")]; OLD = ["Alarm", "Weather", "Music", "Calendar", "Lists", "News", "Transport"]; NEW = "Recipe"; SRC2 = "clinc150"; CALF = a.extra + "/skill_calib.jsonl"
E_ANN, E_TSEL, E_TREP = (ent[:100], ent[100:150], ent[150:300]) if a.task == "news" else (ent[:60], ent[60:100], ent[100:]); B_ANN, B_TEACH = bbc[:40], bbc[40:]
calib = [json.loads(l)["query"] for l in open(a.calib or CALF)]
m, _ = BL.load(a.art, patch=None); tok = AutoTokenizer.from_pretrained(a.art); log("loaded")
def gen(rows): return common.generate(m, tok, [r["text"] for r in rows])
bp = a.out + ".basepreds.json"
if os.path.exists(bp): BASE = json.load(open(bp))
else:
    allr = items + ent + bbc + old; outs = gen(allr); BASE = {r["id"]: o for r, o in zip(allr, outs)}; json.dump(BASE, open(bp, "w"))
log("base preds", len(BASE))
cor = lambda r: BASE[r["id"]] == r["kind"]
NB = [r for r in items if cor(r)]; N_SEL = [r for r in NB if H("fbx-nb|" + r["id"]) % 2 == 0]; N_REP = [r for r in NB if H("fbx-nb|" + r["id"]) % 2 == 1]
ERR = [r for r in old if not cor(r)]; OLD_ANN = [r for r in ERR if H("fbx-o|" + r["id"]) % 3 != 0]; OLD_TEACH = [r for r in ERR if H("fbx-o|" + r["id"]) % 3 == 0][:200]
SHUF = [r for r in old if cor(r)]
log("neigh base-correct", len(NB), "sel", len(N_SEL), "rep", len(N_REP), "old errors", len(ERR), "ann pool", len(OLD_ANN), "old teach", len(OLD_TEACH), "ent base said", collections.Counter(BASE[r["id"]] for r in E_TREP))
cal = FB.calib_seqs(m, tok, calib)
def run_arm(ann_rows, labels, cfg):
    """compute the patch from (rows, labels) on the ORIGINAL model and apply it in place; returns (patch, saved, seconds)"""
    t = time.time()
    A = [dict(query=r["text"], label=l) for r, l in zip(ann_rows, labels)]
    P = FB.compute_patch(m, tok, A, None, calib, cal=cal, **cfg)      # edits the model in place; P["orig"] holds the replaced columns
    return P, P, time.time() - t
def score(rows, outs, labels=None): return [int(o == (labels[i] if labels else r["kind"])) for i, (r, o) in enumerate(zip(rows, outs))]
def evalsets(spec):
    res = {}
    for name, (rows, lab) in spec.items():
        o = gen(rows); res[name] = dict(ok=score(rows, o, [lab] * len(rows) if lab else None), pred=collections.Counter(o).most_common(6))
    return res
def nbytes(P): return 0 if P is None else sum(c.numel() * 2 + v.numel() * 2 for c, v in zip(P["cols"], P["vals"]))
wrongl = lambda r, rnd: rnd.choice([l for l in OLD if l != BASE[r["id"]]])
def draw(n, d, pool): rr = random.Random(1000 * n + d); return rr.sample(pool, n)
def conds(n, d, pool=E_ANN):
    ix = draw(n, d, pool); rr = random.Random(7000 + 1000 * n + d); out = {}
    wl = [wrongl(r, rr) for r in ix]; hm = set(rr.sample(range(n), n // 2))
    out["real"] = (ix, [NEW] * n); out["shuffled"] = (rr.sample(SHUF, n), [NEW] * n); out["wrong"] = (ix, wl); out["mixed"] = (ix, [wl[i] if i in hm else NEW for i in range(n)])
    return out
RES = dict(mode=a.mode, sizes=dict(n_sel=len(N_SEL), n_rep=len(N_REP), old_ann=len(OLD_ANN), old_teach=len(OLD_TEACH), e_tsel=len(E_TSEL), e_trep=len(E_TREP), bbc_teach=len(B_TEACH)),
           base=dict(tsel=float(np.mean([BASE[r["id"]] == NEW for r in E_TSEL])), trep=float(np.mean([BASE[r["id"]] == NEW for r in E_TREP])), bbc=float(np.mean([BASE[r["id"]] == NEW for r in B_TEACH]))), runs=[])
cfgs = json.load(open(a.cfgs)) if a.cfgs else [dict(tag="default")]
def save(): json.dump(RES, open(a.out, "w"))
def finish(tag, arm, n, d, cfg, P, sv, secs, ev, extra=None):
    r = dict(tag=tag, arm=arm, n=n, d=d, cfg={k: v for k, v in cfg.items() if k != "tag"}, bytes=nbytes(P), secs=round(secs, 1), res=ev, **(extra or {})); RES["runs"].append(r); save()
    f = lambda k: round(float(np.mean(ev[k]["ok"])), 3) if k in ev and ev[k]["ok"] else None
    log(tag, arm, "n", n, "d", d, {k: f(k) for k in ev}, "bytes", r["bytes"], f"edit {secs:.0f}s")
if a.mode == "sweep":
    for cfg in cfgs:
        tag = cfg.get("tag", "cfg"); c = {k: v for k, v in cfg.items() if k != "tag"}
        for d in range(2):
            cs = conds(25, d)
            for arm in ("real", "shuffled"):
                rows, lab = cs[arm]; P, sv, secs = run_arm(rows, lab, c)
                ev = evalsets(dict(teach_sel=(E_TSEL, NEW), **({"neigh_sel": (N_SEL, None), "neigh_sel_hp": ([r for r in N_SEL if r["src"] == SRC2], None)} if arm == "real" else {})))
                (FB.undo(m, sv) if sv is not None else None); finish(tag, arm, 25, d, c, P, sv, secs, ev)
elif a.mode == "final":
    c = cfgs[0]; tag = c.get("tag", "final"); c = {k: v for k, v in c.items() if k != "tag"}
    TEACH = dict(teach_rep=(E_TREP, NEW), teach_bbc=(B_TEACH, NEW))
    NEIGH = dict(neigh_rep=(N_REP, None), neigh_sel=(N_SEL, None))
    for n in (25, 12, 5):
        for d in range(a.ndraw if n == 25 else a.ndraw_aux):
            cs = conds(n, d)
            for arm in ("real", "shuffled", "wrong", "mixed"):
                if n != 25 and arm not in ("real", "shuffled"): continue
                rows, lab = cs[arm]; P, sv, secs = run_arm(rows, lab, c)
                spec = dict(TEACH); 
                if arm == "real": spec.update(NEIGH)
                if arm in ("real", "wrong", "mixed") and n == 25: spec["neigh_rep"] = (N_REP, None)
                ev = evalsets(spec); (FB.undo(m, sv) if sv is not None else None); finish(tag, arm, n, d, c, P, sv, secs, ev)
    # empty = no patch: base behaviour by construction (also verified by generation once)
    ev = evalsets(dict(TEACH, neigh_rep=(N_REP, None))); RES["runs"].append(dict(tag=tag, arm="empty", n=0, d=0, res=ev, bytes=0)); save()
    # annotations from source B (BBC entertainment) -> teach on held-out B and on A; calibration stays AG (source A)
    for d in range(a.ndraw_aux):
        ix = draw(25, d, B_ANN); P, sv, secs = run_arm(ix, [NEW] * 25, c)
        tb = [r for r in B_TEACH]; ev = evalsets(dict(teach_rep=(E_TREP, NEW), teach_bbc=(tb, NEW), neigh_rep=(N_REP, None))); (FB.undo(m, sv) if sv is not None else None); finish(tag, "annB", 25, d, c, P, sv, secs, ev)
elif a.mode == "policy":
    c = cfgs[0]; tag = c.get("tag", "policy"); c = {k: v for k, v in c.items() if k != "tag"}
    SPEC = dict(teach_rep=(E_TREP, NEW), neigh_rep=(N_REP, None), old_teach=(OLD_TEACH, None))
    for d in range(a.ndraw_aux):
        rr = random.Random(555 + d)
        arms = {"ent25": ([*draw(25, d, E_ANN)], None), "mix13_12": (None, 13), "mix20_5": (None, 20), "old25": (None, 0)}
        for arm, (_, kent) in arms.items():
            ke = 25 if arm == "ent25" else kent; ent_ix = rr.sample(E_ANN, ke); old_ix = rr.sample(OLD_ANN, 25 - ke)
            rows = ent_ix + old_ix; lab = [NEW] * ke + [r["kind"] for r in old_ix]; P, sv, secs = run_arm(rows, lab, c)
            ev = evalsets(SPEC); (FB.undo(m, sv) if sv is not None else None); finish(tag, arm, 25, d, c, P, sv, secs, ev)
    ev = evalsets(SPEC); RES["runs"].append(dict(tag=tag, arm="empty", n=0, d=0, res=ev, bytes=0))
RES["done"] = True; save(); log("DONE")
