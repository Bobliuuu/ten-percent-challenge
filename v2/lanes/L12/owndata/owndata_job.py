"""L12 own-data dry run (same protocol as L9b REPORT sec 11 / L1 / L5): base Qwen3.5-35B-A3B vs L12 body (mass k25 + RTN4, fake-quant engine; packed tensors are bit-exact vs this path),
(a) no feedback, (b) shared-expert down_proj carrier (nlay20 lam2000 awt30 m64) with 25 Recipe-pool annotations, (c) unseen label Timer (12 annotations x3 draws real/shuffled/wrong + 20-annotation arm).
Sets: Banking77-200 with 77-label menu, same queries under the 7-skill router prompt (agreement), 100 generic prompts, Timer (28 held-out, 60 Alarm + 60 other SEL neighbours)."""
import sys, os, json, argparse, random, collections, csv, io, urllib.request, math, shutil, glob
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
ap = argparse.ArgumentParser(); ap.add_argument("--out", default="/workspace/out_od"); ap.add_argument("--root", default="/workspace/fin"); ap.add_argument("--model", default="Qwen/Qwen3.5-35B-A3B"); ap.add_argument("--cfg", default="best.json")
a = ap.parse_args(); ROOT = os.path.abspath(a.root); sys.path.insert(0, ROOT + "/v2/exam"); sys.path.insert(0, os.path.dirname(HERE))
import torch, loader as L, l12_core as C, l12_fb as FB
from l12_core import log
from _helpers import banking_rows, generic_prompts, timer_sets, BK_URL
L.verify(); T = "SKILL7"; cfg = json.load(open(os.path.join(os.path.dirname(HERE), a.cfg))); os.makedirs(a.out + "/data", exist_ok=True); os.makedirs(a.out + "/res", exist_ok=True)
w = lambda n, rows: open(f"{a.out}/data/{n}.jsonl", "w").write("".join(json.dumps(r) + "\n" for r in rows))
cal = L.load(T, "cal_big"); sel = L.load(T, "sel"); w("cal", cal); w("sel", sel); w("rep", sel[:50])
for p in glob.glob("/r/fbf2/res/stats_*.pt")[:1]: shutil.copy(p, a.out + "/res/")
pool = L.load(T, "ann_pool"); NEW = L.new_label(T); OLD = L.labels(T); RP = L.prompt(T)
assert "{X}" in RP
eng = C.Engine(a.model, a.out + "/data", a.out); eng.max_new = 6; eng.run_stats(len(cal))
ids_of = lambda u: eng.tok(eng.chat(u), add_special_tokens=False).input_ids
def gen(prompts, mx, budget=12000):
    rows = [dict(ids=ids_of(p)) for p in prompts]; return eng.gen_eval(rows, budget=budget, max_new=mx)
def norm(s): return s.strip().lower().split("\n")[0].strip(" .,:;\"'`*") if s.strip() else ""
def wilson(k, n, z=1.96):
    if n == 0: return 0.0
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); r = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)); return (c - r) / d
bk = banking_rows(200); txt = urllib.request.urlopen(BK_URL, timeout=60).read().decode(); menu = sorted({r["category"] for r in csv.DictReader(io.StringIO(txt))}); assert len(menu) == 77
BP = "Classify this banking customer query into exactly one intent. Answer with only the intent name, chosen from: " + ", ".join(menu) + ".\n\nQuery: {t}"
P_bk = [BP.replace("{t}", t) for t, _ in bk]; gold = [c for _, c in bk]; P_rt = [RP.replace("{X}", t) for t, _ in bk[:100]]; P_gen = generic_prompts()[:100]; assert len(P_gen) == 100
def own(): return dict(bk=gen(P_bk, 20), rt=gen(P_rt, 8), gen=gen(P_gen, 24))
alarm = [r for r in sel if r["answer"] == "Alarm"][:60]; others = [r for r in sel if r["answer"] != "Alarm"][::9][:60]
tpool, ttest = timer_sets()
def tev(tag):
    o = gen([RP.replace("{X}", t) for t in ttest], 8); al = gen([r["user"] for r in alarm], 8); x = gen([r["user"] for r in others], 8)
    return dict(tag=tag, timer_emitted=sum(norm(v) == "timer" for v in o) / len(o), alarm_still_alarm=sum(norm(v) == "alarm" for v in al) / len(al), alarm_to_timer=sum(norm(v) == "timer" for v in al) / len(al),
                others_correct=sum(norm(v) == r["answer"].lower() for v, r in zip(x, others)) / len(others), others_to_timer=sum(norm(v) == "timer" for v in x) / len(x), timer_sample=o[:4])
def summarize(OB, O, tag):
    cb = [norm(o) == g.lower() for o, g in zip(OB["bk"], gold)]; cp = [norm(o) == g.lower() for o, g in zip(O["bk"], gold)]; bc = sum(cb); both = sum(1 for x, y in zip(cb, cp) if x and y)
    sm = [x.strip() == y.strip() for x, y in zip(OB["gen"], O["gen"])]; fl = [norm(x) == norm(y) for x, y in zip(OB["gen"], O["gen"])]
    return dict(tag=tag, banking77_200=dict(base_acc=bc / 200, patched_acc=sum(cp) / 200, retention=both / max(bc, 1), retention_wilson_lo=wilson(both, bc), n_base_correct=bc, same_output=sum(norm(x) == norm(y) for x, y in zip(OB["bk"], O["bk"])) / 200),
                router_on_banking_100=dict(same_label=sum(norm(x) == norm(y) for x, y in zip(OB["rt"], O["rt"])) / 100, base_dist=dict(collections.Counter(norm(o) for o in OB["rt"]).most_common(10)), patched_dist=dict(collections.Counter(norm(o) for o in O["rt"]).most_common(10))),
                generic_100=dict(identical_24tok=sum(sm) / 100, same_first_line=sum(fl) / 100),
                diverge=[(P_gen[i], OB["gen"][i], O["gen"][i]) for i in range(100) if not fl[i]][:12], samples_bank=[(gold[i], OB["bk"][i], O["bk"][i]) for i in range(8)])
def short(d): return {k: v for k, v in d.items() if k not in ("diverge", "samples_bank", "timer_sample")}
res = dict(model="Qwen3.5-35B-A3B base vs L12 (mass k25 + RTN4, fake-quant engine)", cfg=cfg, n_banking=200, n_router=100, n_generic=100); outs = {}
def save(): json.dump(dict(res=res, outs=outs), open(a.out + "/owndata_full.json", "w"), indent=1); json.dump(res, open(a.out + "/owndata_summary.json", "w"), indent=1)
# ---- base (unquantised bf16 model, all experts)
eng.set_full_all(); eng.restore_nonexpert(); OB = own(); outs["base"] = OB; res["timer_base"] = tev("base (unquantised)"); log("base done", res["timer_base"]); save()
# ---- L12 body
eng.free_live(); eng.set_full_all(); eng.restore_nonexpert(); eng.free_live(); kept, by, nfb = eng.build(cfg); res["bytes"] = by
OQ = own(); outs["l12"] = OQ; res["no_feedback"] = summarize(OB, OQ, "L12 body, no feedback"); log("NOFB", json.dumps(short(res["no_feedback"])))
res["deterministic_recheck"] = gen(P_bk[:48], 20) == OQ["bk"][:48]; res["timer_l12_empty"] = tev("l12 empty"); log("timer empty", short(res["timer_l12_empty"])); save()
# ---- carrier (L12 REPORT 3c recipe): shared-expert down_proj, nlay 20 lam 2000 awt 30 margin 64, cols 512
exec(open(os.path.join(HERE, "carrier.py")).read())
G_ = dict(nlay=20, lam=2000.0, awt=30.0, margin=64.0, cols=512); hsc = cal_stats(eng.NL - G_["nlay"])
rows_of = lambda u: dict(ids=ids_of(u))
def apply(ann_users, targets):
    return fb_edit_s([rows_of(u) for u in ann_users], targets, G_["nlay"], G_["lam"], G_["awt"], G_["margin"], G_["cols"], hsc=hsc)
# Recipe-pool arm: 25 annotated mistakes of the stock body (draw 100, as the lane's feedback runs), real = Recipe
tkp = [dict(r, ids=ids_of(r["user"])) for r in pool]; mist = [r for r, o in zip(tkp, eng.gen_eval(tkp)) if o != NEW]; rnd = random.Random(100); m2 = mist[:]; rnd.shuffle(m2); ann = m2[:25]
undo, info, cb = fb_edit_s(ann, [NEW] * 25, G_["nlay"], G_["lam"], G_["awt"], G_["margin"], G_["cols"], hsc=hsc); O = own(); tr = tev("recipe25 carrier"); undo()
outs["recipe25"] = O; res["recipe25_carrier"] = summarize(OB, O, "25 Recipe-pool annotations, shared-expert carrier"); res["recipe25_carrier"]["timer_eval"] = tr; res["recipe25_carrier"]["carrier_bytes"] = cb; log("RECIPE25", json.dumps(short(res["recipe25_carrier"]))); save()
res["unpatched_after_recipe"] = dict(own_same=own() == OQ); save()
# Timer arms
def timer_arm(nann, draws, tag):
    out = dict(n_ann=nann, n_pool=len(tpool), n_test=len(ttest), n_alarm_nb=len(alarm), n_other_nb=len(others), draws=[])
    for d in range(draws):
        rng = random.Random(900 + d); an = rng.sample(tpool, nann); dr = dict(draw=d)
        for cond in ("real", "shuffled", "wrong"):
            tg = ["Timer"] * nann
            if cond == "shuffled": tg = ["Alarm" if i % 2 else "Timer" for i in range(nann)]
            if cond == "wrong": tg = [rng.choice(["Weather", "Music", "News", "Lists"]) for _ in an]
            und, inf, cbb = apply([RP.replace("{X}", t) for t in an], tg); dr[cond] = tev(cond)
            if cond == "real" and d == 0 and nann == 12: outs["timer_real_d0"] = own(); res["timer_real_d0_patch"] = summarize(OB, outs["timer_real_d0"], "Timer real draw 0, carrier")
            und(); log("timer", tag, d, cond, {k: v for k, v in dr[cond].items() if k != "timer_sample"}); dr[cond]["carrier_bytes"] = cbb
        out["draws"].append(dr); res[tag] = out; save()
    return out
timer_arm(12, 3, "timer_n12"); timer_arm(20, 1, "timer_n20")
res["unpatched_after"] = dict(timer=tev("after_undo_empty"), own_same=own() == OQ); save()
print("RESULT_JSON", json.dumps(res), flush=True); print("OD_DONE")
