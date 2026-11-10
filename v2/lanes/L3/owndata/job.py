"""L3 own-data job body (see modal_owndata.py). Runs inside the container: base first, then the packed artifact with no feedback, then feedback arms."""
import sys, os, json, time, random, math, gc, shutil, collections, csv, io, urllib.request
import torch
sys.path[:0] = ["/root/l3", "/root/l3/owndata"]
from _helpers import banking_rows, generic_prompts, timer_sets, BK_URL
ART = "/art/skill7x"; MID = "Qwen/Qwen3-30B-A3B-Instruct-2507"; T0 = time.time(); log = lambda *a: print(f"[{time.time()-T0:6.0f}s]", *a, flush=True)
norm = lambda s: (s.strip().split("\n")[0] if s.strip() else "").strip().strip(".,!?:;\"'`* ").lower()
def wilson(k, n, z=1.96):
    if n == 0: return 0.0
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); r = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)); return (c - r) / d
@torch.no_grad()
def gen(tok, m, prompts, mx, bs=25):
    ps = [tok.apply_chat_template([{"role": "user", "content": x}], tokenize=False, add_generation_prompt=True) for x in prompts]
    order = sorted(range(len(ps)), key=lambda i: len(ps[i])); out = [None] * len(ps); tok.padding_side = "left"
    for s0 in range(0, len(order), bs):
        ix = order[s0:s0 + bs]; enc = tok([ps[i] for i in ix], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        g = m.generate(**enc, max_new_tokens=mx, do_sample=False, pad_token_id=tok.pad_token_id)
        for i, o in zip(ix, tok.batch_decode(g[:, enc.input_ids.shape[1]:], skip_special_tokens=True)): out[i] = o.strip()
    return out
def main():
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import runtime, fbl3
    for f in ("task.json", "calib_prompts.jsonl"):
        if not os.path.exists(f"{ART}/{f}"): shutil.copy(f"/root/l3/pkg_skill7x/{f}", f"{ART}/{f}")
    for f in os.listdir(ART):
        if f.startswith("edit_"): os.remove(f"{ART}/{f}")
    T = json.load(open(f"{ART}/task.json")); RP = T["prompt"]; PFX = RP.split("{X}")[0]
    bk = banking_rows(200); gen_p = generic_prompts()[:100]; assert len(gen_p) == 100
    txt = urllib.request.urlopen(BK_URL, timeout=60).read().decode(); menu = sorted({r["category"] for r in csv.DictReader(io.StringIO(txt))}); assert len(menu) == 77
    BP = "Classify this banking customer query into exactly one intent. Answer with only the intent name, chosen from: " + ", ".join(menu) + ".\n\nQuery: {t}"
    P_bk = [BP.replace("{t}", t) for t, _ in bk]; gold = [c for _, c in bk]; P_rt = [RP.replace("{X}", t) for t, _ in bk[:100]]
    rd = lambda p: [json.loads(l) for l in open(p)]
    sel = rd("/root/data/skill7x_sel.jsonl"); alarm = [r for r in sel if r["answer"] == "Alarm"][:60]; others = [r for r in sel if r["answer"] != "Alarm"][::9][:60]
    ux = lambda r: r["user"][len(PFX):]; pool, test = timer_sets()
    def own(tok, m): return dict(bk=gen(tok, m, P_bk, 20), rt=gen(tok, m, P_rt, 8), gen=gen(tok, m, gen_p, 24))
    def tev(tok, m, tag):
        o = gen(tok, m, [RP.replace("{X}", t) for t in test], 8); a = gen(tok, m, [RP.replace("{X}", ux(r)) for r in alarm], 8); x = gen(tok, m, [RP.replace("{X}", ux(r)) for r in others], 8)
        e = dict(tag=tag, timer_emitted=sum(norm(v) == "timer" for v in o) / len(o), alarm_still_alarm=sum(norm(v) == "alarm" for v in a) / len(a), alarm_to_timer=sum(norm(v) == "timer" for v in a) / len(a),
                 others_correct=sum(norm(v) == r["answer"].lower() for v, r in zip(x, others)) / len(others), others_to_timer=sum(norm(v) == "timer" for v in x) / len(x), timer_sample=o[:4]); log("timer", {k: v for k, v in e.items() if k != "timer_sample"}); return e
    # ---- base
    tokb = AutoTokenizer.from_pretrained(MID); base = AutoModelForCausalLM.from_pretrained(MID, torch_dtype=torch.bfloat16, device_map={"": 0}).eval(); log("base loaded")
    OB = own(tokb, base); log("base own done"); TB = tev(tokb, base, "base (no feedback)")
    del base; gc.collect(); torch.cuda.empty_cache()
    # ---- packed
    model, tok, task = runtime.load(ART); log("packed loaded"); OP = own(tok, model); log("packed own done")
    def summarize(O, tag):
        cb = [norm(o) == g.lower() for o, g in zip(OB["bk"], gold)]; cp = [norm(o) == g.lower() for o, g in zip(O["bk"], gold)]; bc = sum(cb); both = sum(1 for a, b in zip(cb, cp) if a and b)
        sm = [a.strip() == b.strip() for a, b in zip(OB["gen"], O["gen"])]; fl = [norm(a) == norm(b) for a, b in zip(OB["gen"], O["gen"])]
        return dict(tag=tag, banking77_200=dict(base_acc=bc / 200, patched_acc=sum(cp) / 200, retention=both / max(bc, 1), retention_wilson_lo=wilson(both, bc), n_base_correct=bc, same_output=sum(norm(a) == norm(b) for a, b in zip(OB["bk"], O["bk"])) / 200),
                    router_on_banking_100=dict(same_label=sum(norm(a) == norm(b) for a, b in zip(OB["rt"], O["rt"])) / 100, base_dist=dict(collections.Counter(norm(o) for o in OB["rt"]).most_common(10)), patched_dist=dict(collections.Counter(norm(o) for o in O["rt"]).most_common(10))),
                    generic_100=dict(identical_24tok=sum(sm) / 100, same_first_line=sum(fl) / 100),
                    diverge=[(gen_p[i], OB["gen"][i], O["gen"][i]) for i in range(100) if not fl[i]][:12], samples_bank=[(gold[i], OB["bk"][i], O["bk"][i]) for i in range(8)])
    res = dict(model="Qwen3-30B-A3B-Instruct-2507 base vs L3 packed skill7x artifact", n_banking=200, n_router=100, n_generic=100, banking_menu=77, base_timer=TB)
    res["nofeedback"] = summarize(OP, "packed, no feedback"); log("nofeedback", json.dumps({k: v for k, v in res["nofeedback"].items() if k not in ("diverge", "samples_bank")}))
    outs = dict(base=OB, packed=OP)
    # ---- feedback machinery (shipped defaults: layers 40-47, prot, lam1000, awt10, m32)
    cfg = dict(layers=list(range(40, 48)), solver="prot", lam=1000.0, awt=10.0, m=32.0); cfg.update(task.get("fb", {}))
    cal_pk = rd(f"{ART}/calib_prompts.jsonl"); calset = {r["text"] for r in cal_pk}; fb = fbl3.FB(model, tok, task, cal_pk, cfg["layers"]); log("fb cov done", cfg)
    errpool = [r for r in rd("/root/data/skill7x_cal.jsonl") if ux(r) not in calset][:600]
    eo = gen(tok, model, [RP.replace("{X}", ux(r)) for r in errpool], 6); old_err = [dict(text=ux(r), label=r["answer"]) for r, o in zip(errpool, eo) if o != r["answer"]]; log("old-skill errors of packed model", len(old_err), "of", len(errpool))
    res["n_old_err_pool"] = len(old_err)
    def with_edit(rows, f):
        try:
            new = fb.edit(rows, layers=cfg["layers"], solver=cfg["solver"], lam=cfg["lam"], awt=cfg["awt"], m=cfg["m"], verbose=False); assert new; return f(), fbl3.edit_bytes(new)
        finally: fb.restore()
    # Recipe carrier (dev annotation pool, never the locked exam)
    ann = sorted(rd("/root/data/skill7x_ann.jsonl"), key=lambda r: r["id"])[:60]; rng = random.Random(100); an = rng.sample(ann, 12); ao = rng.sample(old_err, min(12, len(old_err)))
    rows = [dict(text=ux(r), label="Recipe") for r in an] + ao; O, nb = with_edit(rows, lambda: own(tok, model)); outs["recipe12"] = O
    res["recipe_carrier"] = summarize(O, "12 Recipe + 12 old-skill errors, final carrier"); res["recipe_carrier"]["edit_bytes"] = nb; log("recipe carrier", json.dumps({k: v for k, v in res["recipe_carrier"].items() if k not in ("diverge", "samples_bank")}))
    # Timer
    tout = dict(n_pool=len(pool), n_test=len(test), n_alarm_nb=len(alarm), n_other_nb=len(others), empty=tev(tok, model, "empty (packed, no feedback)"), draws=[])
    for d in range(3):
        rng = random.Random(900 + d); tann = rng.sample(pool, 12); ao = rng.sample(old_err, min(12, len(old_err))); dr = dict(draw=d)
        for cond in ("real", "shuffled", "wrong"):
            nw = [dict(text=t, label="Timer") for t in tann]
            if cond == "real": its = nw + ao
            elif cond == "shuffled": olds = [r["label"] for r in ao]; its = [dict(r, label=olds[i % len(olds)]) for i, r in enumerate(nw)] + [dict(text=r["text"], label="Timer") for r in ao]
            else: its = [dict(r, label=rng.choice(["Weather", "Music", "News", "Lists", "Calendar", "Transport"])) for r in nw + ao]
            def f():
                e = tev(tok, model, f"d{d} {cond}")
                if cond == "real" and d == 0: outs["timer_real_d0"] = own(tok, model)
                return e
            dr[cond], nb = with_edit(its, f); dr[cond]["edit_bytes"] = nb
        tout["draws"].append(dr)
    res["timer_unseen_label"] = tout; res["timer_real_d0_carrier"] = summarize(outs["timer_real_d0"], "Timer real draw 0, final carrier")
    res["unpatched_after"] = dict(timer=tev(tok, model, "after restore"), own_same=own(tok, model) == dict(bk=OP["bk"], rt=OP["rt"], gen=OP["gen"]))
    js = json.dumps(dict(res=res, outs=outs), indent=1); os.makedirs("/art/results", exist_ok=True); open("/art/results/owndata_l3_full.json", "w").write(js); open("/art/results/owndata_l3.json", "w").write(json.dumps(res, indent=1))
    print("RESULT_JSON", json.dumps(res), flush=True); return js
