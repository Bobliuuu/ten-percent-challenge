"""Timer-only arm with a chosen annotation count (default 25 = shipped max), packed L5 model, same sets/controls as owndata_job.run. Pool of Timer templates has 64 rows; test = held-out 28."""
import json, os, sys, time, random
HERE = os.path.dirname(os.path.abspath(__file__)); SHIP = "/root/ship"; sys.path.insert(0, SHIP); sys.path.insert(0, HERE)
from owndata_job import gen, norm
from _helpers import timer_sets
def run(art="/r/pk", out_dir="/r/res_owndata", n_ann=25):
    import common; from packed_loader import load_packed; from fb_core import FB, apply_patch, METHOD
    T0 = time.time(); log = lambda *a: print(f"[{time.time()-T0:6.0f}s]", *a, flush=True)
    T = json.load(open(SHIP + "/skill_task.json")); RP = T["prompt"]; tok, m = load_packed(art, log=lambda *a: None); tok.padding_side = "left"
    DD = json.load(open(SHIP + "/SKILL7.json")); sel = DD["sel"]; pre = RP.split("{t}")[0]; req = lambda r: r["user"][len(pre):]
    pool, test = timer_sets(); alarm = [r for r in sel if r["answer"] == "Alarm"][:60]; others = [r for r in sel if r["answer"] != "Alarm"][::9][:60]
    fb = FB(tok, m, common.prompt_fn(T), common.calib(SHIP + "/feedback_calib.jsonl")); orig = {li: m.model.layers[li].self_attn.o_proj.weight.data.clone() for li in METHOD["layers"]}
    def undo():
        for li, w in orig.items(): m.model.layers[li].self_attn.o_proj.weight.data.copy_(w)
    ask = lambda texts: gen(tok, m, [RP.replace("{t}", t) for t in texts], 8)
    def ev(tag):
        o = ask(test); a = ask([req(r) for r in alarm]); x = ask([req(r) for r in others])
        return dict(tag=tag, timer_emitted=sum(norm(v) == "timer" for v in o) / len(o), alarm_still_alarm=sum(norm(v) == "alarm" for v in a) / len(a), alarm_to_timer=sum(norm(v) == "timer" for v in a) / len(a),
                    others_correct=sum(norm(v) == r["answer"].lower() for v, r in zip(x, others)) / len(others), others_to_timer=sum(norm(v) == "timer" for v in x) / len(x), timer_sample=o[:4])
    res = dict(n_ann=n_ann, cols=METHOD["cols"], n_pool=len(pool), n_test=len(test), empty=ev("empty"), draws=[]); log(res["empty"])
    for d in range(3):
        rng = random.Random(900 + d); ann = rng.sample(pool, min(n_ann, len(pool))); dr = dict(draw=d)
        for cond in ("real", "shuffled", "wrong"):
            its = [(t, "Timer") for t in ann]
            if cond == "shuffled": its = [(t, "Alarm" if i % 2 else "Timer") for i, t in enumerate(ann)]
            if cond == "wrong": its = [(t, rng.choice(["Weather", "Music", "News", "Lists"])) for t in ann]
            patch, n = fb.edit(its, dict(METHOD)); fb.restore(); apply_patch(m, patch); dr[cond] = ev(cond); undo(); log("timer", d, cond, {k: v for k, v in dr[cond].items() if k != "timer_sample"})
        res["draws"].append(dr)
    os.makedirs(out_dir, exist_ok=True); json.dump(res, open(f"{out_dir}/owndata_timer_n{n_ann}.json", "w"), indent=1); print("RESULT_JSON", json.dumps(res), flush=True); return res
