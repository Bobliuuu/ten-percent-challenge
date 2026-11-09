"""L5 own-data dry run (same protocol as L9b REPORT sec 11): base Qwen2.5-32B-Instruct vs packed L5 body (shipped packed loader, no feedback).
Banking77 200 (77-label menu), same queries under the 7-skill router prompt (agreement), 100 generic prompts, unseen label Timer (12 annotations, 3 draws, real/shuffled/wrong).
Run: modal run --detach v2/lanes/L5/owndata/modal_owndata.py   (this file is imported inside the container)."""
import json, os, sys, time, random, csv, io, urllib.request, collections, gc, math
HERE = os.path.dirname(os.path.abspath(__file__)); SHIP = "/root/ship"; sys.path.insert(0, SHIP); sys.path.insert(0, HERE)
import torch
from _helpers import *

def norm(s): return (s.strip().split("\n")[0] if s.strip() else "").strip().strip(".,!?\"'` ").lower()
def wilson(k, n, z=1.96):
    if n == 0: return 0.0
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); r = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)); return (c - r) / d
def gen(tok, m, prompts, max_new, bs=24):
    pids = [tok(tok.apply_chat_template([{"role": "user", "content": x}], tokenize=False, add_generation_prompt=True), add_special_tokens=False).input_ids for x in prompts]
    out = [None] * len(pids); order = sorted(range(len(pids)), key=lambda i: len(pids[i]))
    with torch.no_grad():
        for st in range(0, len(order), bs):
            idx = order[st:st + bs]; L = max(len(pids[i]) for i in idx); ids = torch.full((len(idx), L), tok.pad_token_id); am = torch.zeros((len(idx), L), dtype=torch.long)
            for r, i in enumerate(idx): ids[r, L - len(pids[i]):] = torch.tensor(pids[i]); am[r, L - len(pids[i]):] = 1
            o = m.generate(input_ids=ids.to(m.device), attention_mask=am.to(m.device), max_new_tokens=max_new, do_sample=False, pad_token_id=tok.pad_token_id)
            for r, i in enumerate(idx): out[i] = tok.decode(o[r, L:], skip_special_tokens=True).strip()
    return out
def dist(outs): return dict(collections.Counter(norm(o) for o in outs).most_common(10))

def run(art="/r/pk", out_dir="/r/res"):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    T0 = time.time(); log = lambda *a: print(f"[{time.time()-T0:6.0f}s]", *a, flush=True)
    os.makedirs(out_dir, exist_ok=True)
    bk = banking_rows(200); gen_p = generic_prompts()[:100]
    txt = urllib.request.urlopen(BK_URL, timeout=60).read().decode(); menu = sorted({r["category"] for r in csv.DictReader(io.StringIO(txt))}); log("banking labels", len(menu))
    BP = "Classify this banking customer query into exactly one intent. Answer with only the intent name, chosen from: " + ", ".join(menu) + ".\n\nQuery: {t}"
    T = json.load(open(SHIP + "/skill_task.json")); RP = T["prompt"]
    P_bk = [BP.replace("{t}", t) for t, _ in bk]; P_rt = [RP.replace("{t}", t) for t, _ in bk[:100]]
    def run_all(tok, m, tag):
        o = dict(bk=gen(tok, m, P_bk, 16), rt=gen(tok, m, P_rt, 8)); log(tag, "bk,rt done"); o["gen"] = gen(tok, m, gen_p, 24); log(tag, "gen done"); return o
    ng = torch.cuda.device_count(); mid = "Qwen/Qwen2.5-32B-Instruct"
    tokb = AutoTokenizer.from_pretrained(mid); tokb.padding_side = "left"
    base = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.bfloat16, device_map="auto", max_memory={i: "70GiB" for i in range(ng)}).eval(); log("base loaded")
    OB = run_all(tokb, base, "base"); del base; gc.collect(); torch.cuda.empty_cache()
    import common; from packed_loader import load_packed; from fb_core import FB, apply_patch, METHOD
    tok, m = load_packed(art, log=log); tok.padding_side = "left"; log("packed loaded"); OP = run_all(tok, m, "packed")
    res = dict(model="Qwen2.5-32B-Instruct base vs L5 packed (no feedback)", n_banking=200, n_router=100, n_generic=100, banking_menu=len(menu))
    cb = [norm(o) == c.lower() for o, (_, c) in zip(OB["bk"], bk)]; cp = [norm(o) == c.lower() for o, (_, c) in zip(OP["bk"], bk)]
    bc = sum(cb); both = sum(1 for a, b in zip(cb, cp) if a and b)
    res["banking77_200"] = dict(base_acc=bc / 200, packed_acc=sum(cp) / 200, retention=both / max(bc, 1), retention_wilson_lo=wilson(both, bc), n_base_correct=bc, same_output=sum(norm(a) == norm(b) for a, b in zip(OB["bk"], OP["bk"])) / 200)
    res["router_on_banking_100"] = dict(same_label=sum(norm(a) == norm(b) for a, b in zip(OB["rt"], OP["rt"])) / 100, base_dist=dist(OB["rt"]), packed_dist=dist(OP["rt"]))
    sm = [a.strip() == b.strip() for a, b in zip(OB["gen"], OP["gen"])]; fl = [norm(a) == norm(b) for a, b in zip(OB["gen"], OP["gen"])]
    res["generic_100"] = dict(identical_24tok=sum(sm) / 100, same_first_line=sum(fl) / 100)
    res["samples"] = dict(banking=[(bk[i][1], OB["bk"][i], OP["bk"][i]) for i in range(8)], generic=[(gen_p[i], OB["gen"][i], OP["gen"][i]) for i in range(0, 100, 10)], diverge=[(gen_p[i], OB["gen"][i], OP["gen"][i]) for i in range(100) if not fl[i]][:12])
    log("own-data", json.dumps({k: v for k, v in res.items() if k != "samples"}))
    json.dump(dict(res=res, outs=dict(base=OB, packed=OP)), open(out_dir + "/owndata_s7_L5.json", "w")); json.dump(res, open(out_dir + "/owndata_summary.json", "w"), indent=1)
    # ---- unseen label Timer
    DD = json.load(open(SHIP + "/SKILL7.json")); sel = DD["sel"]; pre = RP.split("{t}")[0]; req = lambda r: r["user"][len(pre):]
    pool, test = timer_sets(); alarm = [r for r in sel if r["answer"] == "Alarm"][:60]; others = [r for r in sel if r["answer"] != "Alarm"][::9][:60]
    pf = common.prompt_fn(T); fb = FB(tok, m, pf, common.calib(SHIP + "/feedback_calib.jsonl")); log("FB ready")
    orig = {li: m.model.layers[li].self_attn.o_proj.weight.data.clone() for li in METHOD["layers"]}
    def undo():
        for li, w in orig.items(): m.model.layers[li].self_attn.o_proj.weight.data.copy_(w)
    ask = lambda texts: gen(tok, m, [RP.replace("{t}", t) for t in texts], 8)
    def ev(tag):
        o = ask(test); a = ask([req(r) for r in alarm]); x = ask([req(r) for r in others])
        return dict(tag=tag, timer_emitted=sum(norm(v) == "timer" for v in o) / len(o), alarm_still_alarm=sum(norm(v) == "alarm" for v in a) / len(a), alarm_to_timer=sum(norm(v) == "timer" for v in a) / len(a),
                    others_correct=sum(norm(v) == r["answer"].lower() for v, r in zip(x, others)) / len(others), others_to_timer=sum(norm(v) == "timer" for v in x) / len(x), timer_sample=o[:4])
    tout = dict(cols=METHOD["cols"], n_pool=len(pool), n_test=len(test), n_alarm_nb=len(alarm), n_other_nb=len(others), empty=ev("empty"), draws=[]); log("timer empty", tout["empty"])
    for d in range(3):
        rng = random.Random(900 + d); ann = rng.sample(pool, 12); dr = dict(draw=d)
        for cond in ("real", "shuffled", "wrong"):
            its = [(t, "Timer") for t in ann]
            if cond == "shuffled": its = [(t, "Alarm" if i % 2 else "Timer") for i, t in enumerate(ann)]
            if cond == "wrong": its = [(t, rng.choice(["Weather", "Music", "News", "Lists"])) for t in ann]
            patch, n = fb.edit(its, dict(METHOD)); fb.restore(); apply_patch(m, patch); dr[cond] = ev(cond); undo(); log("timer", d, cond, {k: v for k, v in dr[cond].items() if k != "timer_sample"})
        tout["draws"].append(dr)
    res["timer_unseen_label"] = tout; json.dump(res, open(out_dir + "/owndata_summary.json", "w"), indent=1); print("RESULT_JSON", json.dumps(res), flush=True); return res
