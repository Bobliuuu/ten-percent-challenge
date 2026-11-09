"""L1 own-data dry run (same protocol as L9b final_jobs.own_data; ENTRY_DECISION sec 3 item 1). Base = released Bonsai-8B (artifact without patch.pt = bit-exact repack).
Patched arms = shipped feedback.compute_patch (default `scale` carrier, in-process, exactly what feedback.py does): (a) 12 Recipe annotations, (b) 12 Timer annotations (3 draws; real/shuffled/wrong).
Own-data sets (Banking77 77-label menu 200, same queries under the shipped 7-skill router prompt 100, generic 100) are scored for base and for the Recipe patch and for Timer-real draw 0.
Timer test: 28 held-out Timer requests, 60 Alarm + 60 other SKILL7 neighbours. Writes results/owndata_summary.json.
  modal run --detach v2/lanes/L1/owndata/modal_owndata.py"""
import modal, os
R = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", ".."))
app = modal.App("tenpct-l1-owndata"); vol = modal.Volume.from_name("tenpct-l1-art")
img = (modal.Image.debian_slim(python_version="3.11").pip_install("torch==2.5.1", "transformers==4.56.2", "numpy", "accelerate")
       .add_local_dir(R + "/v2/lanes/L1", "/repo/L1", ignore=["**/__pycache__", "results/**", "build/**"])
       .add_local_file(R + "/v2/lanes/L9b/final_jobs.py", "/repo/L9b/final_jobs.py").add_local_file(R + "/v2/lanes/L9b/data/skill7_sel.jsonl", "/repo/L9b/skill7_sel.jsonl"))
REC = [("how long do I boil an egg", "Recipe"), ("what is a good substitute for butter in cookies", "Recipe"), ("give me a recipe for vegetable soup", "Recipe"), ("how do i make pancakes from scratch", "Recipe"),
       ("how much flour for one loaf of bread", "Recipe"), ("tell me how to cook rice on the stove", "Recipe"), ("what temperature do I bake chicken at", "Recipe"), ("steps to prepare a tomato sauce", "Recipe"),
       ("how do i marinate steak overnight", "Recipe"), ("what can i use instead of eggs in a cake", "Recipe"), ("recipe for homemade guacamole", "Recipe"), ("how long to simmer a beef stew", "Recipe")]
@app.function(image=img, gpu="L40S", timeout=6 * 3600, volumes={"/vol": vol})
def run():
    import sys, json, random, time, shutil, collections, math, torch
    sys.path.insert(0, "/repo/L9b"); import final_jobs as FJ
    shutil.copytree("/vol/art", "/work/art")
    for f in ("answer.py", "feedback.py", "bonsai_loader.py", "common.py", "skill_task.json", "feedback_calib.jsonl", "tensor_manifest.json"): shutil.copy("/repo/L1/" + f, "/work/art/" + f)
    sys.path.insert(0, "/work/art"); import importlib; importlib.invalidate_caches()
    import bonsai_loader as BL, common, feedback as FB
    from transformers import AutoTokenizer
    assert not os.path.exists("/work/art/patch.pt")
    t0 = time.time(); log = lambda *a: print(f"[{time.time()-t0:6.0f}s]", *a, flush=True)
    norm = lambda s: s.strip().lower().split("\n")[0].strip(" .,:;\"'`*") if s.strip() else ""
    def wilson(k, n, z=1.96):
        if n == 0: return 0.0
        p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); r = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)); return (c - r) / d
    m, _ = BL.load("/work/art", patch=None); tok = AutoTokenizer.from_pretrained("/work/art"); T = common.task("/work/art/skill_task.json")
    cal = [common.prompt_of(json.loads(l), T) for l in open("/work/art/feedback_calib.jsonl") if l.strip()]
    gen = lambda P, mx: common.generate(m, tok, P, mx=mx)
    bk = FB_bk = FJ.banking_rows(200); import urllib.request, csv, io
    txt = urllib.request.urlopen(FJ.BK_URL, timeout=60).read().decode(); menu = sorted({r["category"] for r in csv.DictReader(io.StringIO(txt))}); assert len(menu) == 77
    BP = "Classify this banking customer query into exactly one intent. Answer with only the intent name, chosen from: " + ", ".join(menu) + ".\n\nQuery: {t}"
    P_bk = [BP.replace("{t}", t) for t, _ in bk]; gold = [c for _, c in bk]; P_rt = [T["prompt"].replace("{t}", t) for t, _ in bk[:100]]; P_gen = FJ.generic_prompts()[:100]; assert len(P_gen) == 100
    def own():
        return dict(bk=gen(P_bk, 20), rt=gen(P_rt, 8), gen=gen(P_gen, 24))
    OB = own(); log("base own done")
    # sanity: empty feedback = base (no patch applied -> identical by construction; re-generate banking to check determinism)
    det = gen(P_bk[:48], 20) == OB["bk"][:48]
    sel = [json.loads(l) for l in open("/repo/L9b/skill7_sel.jsonl")]; alarm = [r for r in sel if r["answer"] == "Alarm"][:60]; others = [r for r in sel if r["answer"] != "Alarm"][::9][:60]
    pool, test = FJ.timer_sets(); mk = lambda t: {"text": t, "label": "Timer"}
    def tev(tag):
        o = gen([T["prompt"].replace("{t}", t) for t in test], 8); a = gen([r["user"] for r in alarm], 8); x = gen([r["user"] for r in others], 8)
        return dict(tag=tag, timer_emitted=sum(norm(v) == "timer" for v in o) / len(o), alarm_still_alarm=sum(norm(v) == "alarm" for v in a) / len(a), alarm_to_timer=sum(norm(v) == "timer" for v in a) / len(a),
                    others_correct=sum(norm(v) == r["answer"].lower() for v, r in zip(x, others)) / len(others), others_to_timer=sum(norm(v) == "timer" for v in x) / len(x), timer_sample=o[:4])
    res = dict(n_banking=200, n_router=100, n_generic=100, banking_menu=77, deterministic_recheck=det)
    def summarize(O, tag):
        cb = [norm(o) == g.lower() for o, g in zip(OB["bk"], gold)]; cp = [norm(o) == g.lower() for o, g in zip(O["bk"], gold)]; bc = sum(cb); both = sum(1 for a, b in zip(cb, cp) if a and b)
        sm = [a.strip() == b.strip() for a, b in zip(OB["gen"], O["gen"])]; fl = [norm(a) == norm(b) for a, b in zip(OB["gen"], O["gen"])]
        return dict(tag=tag, banking77_200=dict(base_acc=bc / 200, patched_acc=sum(cp) / 200, retention=both / max(bc, 1), retention_wilson_lo=wilson(both, bc), n_base_correct=bc, same_output=sum(norm(a) == norm(b) for a, b in zip(OB["bk"], O["bk"])) / 200),
                    router_on_banking_100=dict(same_label=sum(norm(a) == norm(b) for a, b in zip(OB["rt"], O["rt"])) / 100, base_dist=dict(collections.Counter(norm(o) for o in OB["rt"]).most_common(10)), patched_dist=dict(collections.Counter(norm(o) for o in O["rt"]).most_common(10))),
                    generic_100=dict(identical_24tok=sum(sm) / 100, same_first_line=sum(fl) / 100),
                    diverge=[(P_gen[i], OB["gen"][i], O["gen"][i]) for i in range(100) if not fl[i]][:12], samples_bank=[(gold[i], OB["bk"][i], O["bk"][i]) for i in range(8)])
    res["base_vs_nofeedback_artifact"] = summarize(OB, "no feedback (patch absent): same weights, same generations")
    outs = dict(base=OB)
    def with_patch(ann, f):
        P = FB.compute_patch(m, tok, ann, T, cal, carrier="scale"); assert P is not None
        try: return f()
        finally: FB.undo(m, P)
    # Recipe patch
    def f_rec(): return own()
    O = with_patch([dict(text=t, label=l) for t, l in REC], f_rec); outs["recipe12"] = O; res["recipe12_patch"] = summarize(O, "12 Recipe annotations, scale carrier"); log("recipe", json.dumps({k: v for k, v in res["recipe12_patch"].items() if k not in ("diverge", "samples_bank")}))
    tout = dict(n_pool=len(pool), n_test=len(test), n_alarm_nb=len(alarm), n_other_nb=len(others), empty=tev("empty"), draws=[]); log("timer empty", {k: v for k, v in tout["empty"].items() if k != "timer_sample"})
    for d in range(3):
        rng = random.Random(900 + d); ann = [mk(t) for t in rng.sample(pool, 12)]; dr = dict(draw=d)
        for cond in ("real", "shuffled", "wrong"):
            its = [dict(r) for r in ann]
            if cond == "shuffled": its = [dict(r, label=("Alarm" if i % 2 else "Timer")) for i, r in enumerate(its)]
            if cond == "wrong": its = [dict(r, label=rng.choice(["Weather", "Music", "News", "Lists"])) for r in its]
            def f():
                e = tev(cond)
                if cond == "real" and d == 0: outs["timer_real_d0"] = own()
                return e
            dr[cond] = with_patch(its, f); log("timer", d, cond, {k: v for k, v in dr[cond].items() if k != "timer_sample"})
        tout["draws"].append(dr)
    res["timer_unseen_label"] = tout; res["timer_real_d0_patch"] = summarize(outs["timer_real_d0"], "Timer real draw 0, scale carrier")
    res["unpatched_after"] = dict(timer=tev("after_undo_empty"), own_same=own() == OB)
    open("/work/res.json", "w").write(json.dumps(dict(res=res, outs=outs), indent=1)); print("RESULT_JSON", json.dumps(res), flush=True)
    return json.dumps(dict(res=res, outs=outs), indent=1)
@app.local_entrypoint()
def main():
    r = run.remote(); d = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results"); open(os.path.join(d, "owndata_full.json"), "w").write(r)
    import json; json.dump(json.loads(r)["res"], open(os.path.join(d, "owndata_summary.json"), "w"), indent=1); print(r[:2000])
