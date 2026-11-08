"""L1+ DEV/SELECTION experiments (SKILL7, Bonsai-8B). Real greedy generation, exact label match. Nothing here reads the sealed exam; the locked TEACH and NEIGHBOUR sets are read only in mode `locked`.
  python run_l1p.py --art ART --extra EXTRA --out OUT.json --mode sweep|final|continual|proxy|robust|locked --cfgs cfgs.json [--ndraw N]
Data: same dev pools and splits as v2/bonsai/fbx/run_fbx.py (SKILL7 branch): A = MASSIVE cooking (ann 60 / teach_sel 40 / teach_rep 45), B = CLINC cooking (ann 40 / teach 444), neighbours = base-correct dev items (sel/rep halves)."""
import sys, os, json, hashlib, random, time, argparse, collections, copy
HERE = os.path.dirname(os.path.abspath(__file__)); SHIP = HERE + "/../ship"; sys.path.insert(0, SHIP)
import torch, numpy as np
import bonsai_loader as BL, common, feedback as FB
from transformers import AutoTokenizer
ap = argparse.ArgumentParser(); ap.add_argument("--art", required=True); ap.add_argument("--extra", required=True); ap.add_argument("--out", required=True); ap.add_argument("--mode", default="sweep")
ap.add_argument("--cfgs"); ap.add_argument("--ndraw", type=int, default=5); ap.add_argument("--root", default=os.path.abspath(HERE + "/../../.."))
a = ap.parse_args(); T0 = time.time(); log = lambda *x: print(f"[{time.time()-T0:6.0f}s]", *x, flush=True)
H = lambda s: int(hashlib.sha256(s.encode()).hexdigest()[:8], 16); rd = lambda p: [json.loads(l) for l in open(p) if l.strip()]
PRO = HERE + "/proxy/"
TPL = json.load(open(HERE + "/../../task_search/EASY/PLAN.json"))["SKILL7"]["prompt"]            # '{X}' template (identical to the shipped task.json prompt with {t})
TASK = {"prompt": TPL.replace("{X}", "{t}")}
def norm(r, src=None): return dict(id=r["id"], text=r["user"], kind=r["kind"], src=r.get("src") or src)
items = [norm(r) for r in rd(HERE + "/../../task_search/EASY/items_dev.jsonl") if r["task"] == "SKILL7"]
ent = sorted([norm(r) for r in rd(a.extra + "/skill_new_a.jsonl")], key=lambda r: H("fbx-ent|" + r["id"])); bbc = sorted([norm(r) for r in rd(a.extra + "/skill_new_b.jsonl")], key=lambda r: H("fbx-bbc|" + r["id"]))
old = [norm(r) for r in rd(a.extra + "/skill_old.jsonl")]; OLD = ["Alarm", "Weather", "Music", "Calendar", "Lists", "News", "Transport"]; NEW = "Recipe"
E_ANN, E_TSEL, E_TREP = ent[:60], ent[60:100], ent[100:]; B_ANN, B_TEACH = bbc[:40], bbc[40:]
calib = [json.loads(l)["query"] for l in open(a.extra + "/skill_calib.jsonl")]
m, _ = BL.load(a.art, patch=None); tok = AutoTokenizer.from_pretrained(a.art); log("loaded")
def gen(rows, key="text"): return common.generate(m, tok, [r[key] for r in rows])
bp = a.out + ".basepreds.json"; extra_sets = {}
for nm in ("email", "takeaway"):
    extra_sets[nm] = sorted([norm(r) for r in rd(a.extra + f"/skill_{nm}.jsonl")], key=lambda r: H("l1p-" + r["id"]))
if os.path.exists(bp): BASE = json.load(open(bp))
else:
    allr = items + ent + bbc + old + extra_sets["email"] + extra_sets["takeaway"]; BASE = {r["id"]: o for r, o in zip(allr, gen(allr))}; json.dump(BASE, open(bp, "w"))
cor = lambda r: BASE[r["id"]] == r["kind"]
NB = [r for r in items if cor(r)]; N_SEL = [r for r in NB if H("fbx-nb|" + r["id"]) % 2 == 0]; N_REP = [r for r in NB if H("fbx-nb|" + r["id"]) % 2 == 1]
SHUF = [r for r in old if cor(r)]; log("neigh", len(N_SEL), len(N_REP), "base said on teach_rep", collections.Counter(BASE[r["id"]] for r in E_TREP).most_common(3))
cal = FB.calib_seqs(m, tok, calib)
def bytes_of(P): return FB.patch_bytes(P)
def relnorm(P, base_w):
    return None
def run_arm(rows, labels, cfg, aug_rows=True):
    A = [dict(text=r["text"].split("Request: ", 1)[1] if "Request: " in r["text"] else r["text"], label=l) for r, l in zip(rows, labels)]
    c = {k: v for k, v in cfg.items() if k not in ("tag",)}; t = time.time()
    return FB.compute_patch(m, tok, A, TASK, calib, cal=cal, **c), time.time() - t
def ok(rows, outs, lab=None): return [int(o == (lab or r["kind"])) for r, o in zip(rows, outs)]
def ev(spec):
    res = {}
    for name, (rows, lab) in spec.items():
        o = gen(rows); res[name] = dict(ok=ok(rows, o, lab), pred=collections.Counter(o).most_common(5), off=sum(x not in OLD + [NEW] for x in o))
    return res
mean = lambda ev_, k: round(float(np.mean(ev_[k]["ok"])), 4) if k in ev_ and ev_[k]["ok"] else None
def draw(n, d, pool): return random.Random(1000 * n + d).sample(pool, n)
RES = dict(mode=a.mode, base_teach_rep=float(np.mean([BASE[r["id"]] == NEW for r in E_TREP])), runs=[]); cfgs = json.load(open(a.cfgs)) if a.cfgs else [dict(tag="default")]
def save(): json.dump(RES, open(a.out, "w"))
def rec(tag, arm, n, d, cfg, P, secs, e, extra=None):
    r = dict(tag=tag, arm=arm, n=n, d=d, cfg={k: v for k, v in cfg.items() if k != "tag"}, bytes=bytes_of(P), secs=round(secs, 1), res=e, **(extra or {})); RES["runs"].append(r); save()
    log(tag, arm, "n", n, "d", d, {k: mean(e, k) for k in e}, "bytes", r["bytes"], f"{secs:.0f}s")
def shuf_rows(n, d): return random.Random(7000 + n * 1000 + d).sample(SHUF, n)

if a.mode == "sweep":      # config choice on SELECTION data only: teach_sel (A), neighbours sel
    for cfg in cfgs:
        for d in range(a.ndraw):
            for arm in ("real", "shuffled"):
                rows = draw(25, d, E_ANN) if arm == "real" else shuf_rows(25, d); P, secs = run_arm(rows, [NEW] * 25, cfg)
                spec = dict(teach_sel=(E_TSEL, NEW)); 
                if arm == "real": spec["neigh_sel"] = (N_SEL, None)
                e = ev(spec); FB.undo(m, P); rec(cfg.get("tag", "cfg"), arm, 25, d, cfg, P, secs, e)
elif a.mode == "final":    # n=25/12/5 x draws, real + shuffled, report sets
    for cfg in cfgs:
        for n in (25, 12, 5):
            for d in range(a.ndraw if n == 25 else max(2, a.ndraw // 2)):
                for arm in ("real", "shuffled"):
                    if arm == "shuffled" and n != 25: continue
                    rows = draw(n, d, E_ANN) if arm == "real" else shuf_rows(n, d); P, secs = run_arm(rows, [NEW] * n, cfg)
                    spec = dict(teach_rep=(E_TREP, NEW), teach_bbc=(B_TEACH, NEW)); 
                    if arm == "real": spec["neigh_rep"] = (N_REP, None)
                    e = ev(spec); FB.undo(m, P); rec(cfg.get("tag", "cfg"), arm, n, d, cfg, P, secs, e)
elif a.mode == "continual":
    SK = {"Recipe": (E_ANN, E_TREP), "Email": (extra_sets["email"][:60], extra_sets["email"][60:260]), "Takeaway": (extra_sets["takeaway"][:50], extra_sets["takeaway"][50:])}
    order = ["Recipe", "Email", "Takeaway"]
    def evall(tag): return ev({**{f"teach_{k}": (SK[k][1], k) for k in order}, "neigh_rep": (N_REP, None)})
    for cfg in cfgs:
        per = cfg.get("per_round", 25); c = {k: v for k, v in cfg.items() if k not in ("tag", "per_round")}
        for d in range(a.ndraw):
            Ps = []; merged = None
            for r_i, k in enumerate(order):
                rr = random.Random(31 * d + r_i); rows = rr.sample(SK[k][0], per); t = time.time()
                A = [dict(text=x["text"].split("Request: ", 1)[1], label=k) for x in rows]; cal_i = FB.calib_seqs(m, tok, calib) if r_i else cal
                P = FB.compute_patch(m, tok, A, TASK, calib, cal=cal_i, **c); Ps.append(P); merged = FB.merge_patch(merged, {kk: vv for kk, vv in P.items() if kk not in ("orig", "orig_w")}) if P else merged
                e = evall(k); rec(cfg.get("tag", "cfg"), f"after_round{r_i+1}_{k}", per, d, cfg, merged, time.time() - t, e)
            for P in reversed(Ps): FB.undo(m, P)
elif a.mode == "proxy":
    P_old = rd(PRO + "old_skills.jsonl"); PN = {k: rd(PRO + f"new_{k}.jsonl") for k in ("translate", "banking", "email")}
    mk = lambda r: dict(id=r["id"], text=TPL.replace("{X}", r["text"]), kind=r["kind"], style=r["style"], raw=r["text"])
    P_old = [mk(r) for r in P_old]; PNm = {k: [mk(r) for r in v] for k, v in PN.items()}; allp = P_old + [x for v in PNm.values() for x in v]
    PB = dict(zip([r["id"] for r in allp], gen(allp))); RES["proxy_base"] = dict(old_acc=float(np.mean([PB[r["id"]] == r["kind"] for r in P_old])), old_off=sum(PB[r["id"]] not in OLD for r in P_old),
        new_said={k: collections.Counter(PB[r["id"]] for r in v).most_common(4) for k, v in PNm.items()}); save(); log("proxy base", RES["proxy_base"])
    PNB = [r for r in P_old if PB[r["id"]] == r["kind"]]; half = lambda r: H("pxcal|" + r["id"]) % 2
    PN_cal = [r for r in PNB if half(r) == 0]; PN_ev = [r for r in PNB if half(r) == 1]; styles = sorted({r["style"] for r in P_old})
    for cfg in cfgs:
        for newk, label in (("translate", "Translate"), ("banking", "Banking"), ("email", "Email")):
            rows_all = sorted(PNm[newk], key=lambda r: H("pxa|" + r["id"])); pool, teach = rows_all[:40], rows_all[40:]
            for d in range(a.ndraw):
                for calmode in ("shipped", "own_unlabeled"):
                    ix = random.Random(900 + d).sample(pool, 25); rows = ix
                    cal_use = cal if calmode == "shipped" else FB.calib_seqs(m, tok, [r["text"] for r in PN_cal])
                    A = [dict(text=r["raw"], label=label) for r in rows]; c = {k: v for k, v in cfg.items() if k != "tag"}; t = time.time()
                    P = FB.compute_patch(m, tok, A, TASK, calib, cal=cal_use, **c)
                    spec = {"teach_" + newk: (teach, label), "old_proxy_ev": (PN_ev, None), "old_dev_rep": (N_REP, None), **{"old_proxy_" + s: ([r for r in PN_ev if r["style"] == s], None) for s in styles}, "dev_recipe": (E_TREP, None)}
                    e = ev({k: v for k, v in spec.items() if k != "dev_recipe"}); FB.undo(m, P); rec(cfg.get("tag", "cfg"), f"{newk}_{calmode}", 25, d, cfg, P, time.time() - t, e)
            # shuffled / wrong controls once
            ix = random.Random(900).sample(pool, 25); c = {k: v for k, v in cfg.items() if k != "tag"}
            sh = random.Random(77).sample(PN_cal, 25); P = FB.compute_patch(m, tok, [dict(text=r["raw"], label=label) for r in sh], TASK, calib, cal=cal, **c)
            e = ev({"teach_" + newk: (teach, label), "old_proxy_ev": (PN_ev, None)}); FB.undo(m, P); rec(cfg.get("tag", "cfg"), f"{newk}_shuffled", 25, 0, cfg, P, 0, e)
elif a.mode == "robust":
    VAR = {
     "orig": lambda t: TPL.replace("{X}", t),
     "menu_reordered": lambda t: "You are the router of a voice assistant. Decide which skill should handle the user's request.\nAnswer with exactly one word from: Transport, News, Lists, Calendar, Music, Weather, Alarm.\n\nRequest: " + t,
     "paraphrased_instr": lambda t: "Which skill of the voice assistant should take care of this? Reply with a single word chosen from Alarm, Weather, Music, Calendar, Lists, News, Transport.\n\nUser said: " + t,
     "no_menu": lambda t: "Route this voice request to the right skill. Reply with one word.\n\nRequest: " + t,
     "markdown_wrapper": lambda t: "### Task\nClassify the request into a skill (Alarm, Weather, Music, Calendar, Lists, News, Transport). Output the skill name only.\n### Input\n" + t + "\n### Skill",
     "lowercase_text": lambda t: TPL.replace("{X}", t.lower()),
     "long_600": lambda t: TPL.replace("{X}", ("Thanks for earlier, that was helpful and I have been busy at work all week with the move. " * 6) + t),
     "long_3000": lambda t: TPL.replace("{X}", ("So anyway, my cousin was telling me about the new office and how everything has changed since spring, which is a long story. " * 24) + t)}
    txt = lambda r: r["text"].split("Request: ", 1)[1]; OODR = [dict(text=r["text"], kind="OOD") for r in rd(PRO + "new_banking.jsonl")[:80]]
    def evvars(tag):
        out = {}
        for vn, vf in VAR.items():
            mk = lambda rows: [dict(text=vf(txt(r)) if "Request: " in r["text"] else vf(r["text"]), kind=r["kind"]) for r in rows]
            out.update({f"{vn}|{k}": v for k, v in ev({"neigh": (mk(N_REP[:150]), None), "teach": (mk(E_TREP), NEW), "ood": (mk(OODR), "__none__")}).items()})
        return out
    base_e = evvars("base"); RES["base_variants"] = {k: dict(acc=float(np.mean(v["ok"])), off=v["off"], pred=v["pred"]) for k, v in base_e.items()}; save()
    for cfg in cfgs:
        for d in range(a.ndraw):
            P, secs = run_arm(draw(25, d, E_ANN), [NEW] * 25, cfg); e = evvars("patched"); FB.undo(m, P)
            for vn in VAR: rec(cfg.get("tag", "cfg"), "var_" + vn, 25, d, cfg, P, secs, {k.split("|")[1]: v for k, v in e.items() if k.startswith(vn + "|")}, dict(variant=vn))
elif a.mode == "locked":    # locked TEACH + NEIGHBOUR sets only (exam items outside the neighbour list are NOT scored)
    sys.path.insert(0, a.root + "/v2/exam"); import loader as LD; LD.verify()
    teach = LD.load("SKILL7", "teach"); pool = LD.load("SKILL7", "ann_pool"); nid = set(LD.load("SKILL7", "neighbours")); ex = {r["id"]: r for r in LD.load("SKILL7", "exam")}
    neigh_all = [ex[i] for i in nid]; nb_out = gen(neigh_all, "user"); neigh = [r for r, o in zip(neigh_all, nb_out) if o == r["kind"]]; log("locked neigh base-correct", len(neigh), "of", len(neigh_all))
    tx = lambda r: r["user"].split("Request: ", 1)[1]
    sel = LD.load("SKILL7", "sel")[:300]; so = gen(sel, "user"); sel_ok = [r for r, o in zip(sel, so) if o == r["kind"]]; log("shuffled pool (selection items, base-correct)", len(sel_ok))
    for cfg in cfgs:
        for d in range(a.ndraw):
            for arm in ("real", "shuffled"):
                rr = random.Random(2500 + d); ix = rr.sample(pool, 25); sh = rr.sample(sel_ok, 25) if arm == "shuffled" else None
                rows = ix if arm == "real" else sh; c = {k: v for k, v in cfg.items() if k != "tag"}; t = time.time()
                P = FB.compute_patch(m, tok, [dict(text=tx(r), label=NEW) for r in rows], TASK, calib, cal=cal, **c)
                tout = gen(teach, "user"); tk = ok(teach, tout, NEW); e = dict(teach=dict(ok=tk, pred=collections.Counter(tout).most_common(4)))
                if arm == "real": o = gen(neigh, "user"); e["neigh"] = dict(ok=ok(neigh, o), pred=[])
                FB.undo(m, P); bs = collections.defaultdict(list)
                for r, x in zip(teach, tk): bs[r["src"]].append(x)
                rec(cfg.get("tag", "cfg"), arm, 25, d, cfg, P, time.time() - t, e, dict(teach_by_src={k: [round(float(np.mean(v)), 3), len(v)] for k, v in bs.items()}))
RES["done"] = True; save(); log("DONE")
