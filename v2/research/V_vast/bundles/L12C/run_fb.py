"""build a config and run the feedback experiment: python run_fb.py --cfg best.json --out out"""
import json, argparse, torch
import l12_core as C, l12_fb as FB
ap = argparse.ArgumentParser(); ap.add_argument("--cfg", required=True); ap.add_argument("--out", default="out"); ap.add_argument("--data", default="data"); ap.add_argument("--model", default="Qwen/Qwen3-30B-A3B-Instruct-2507")
ap.add_argument("--nlay", type=int, default=4); ap.add_argument("--lam", type=float, default=1.0); ap.add_argument("--margin", type=float, default=2.0); ap.add_argument("--requant", type=int, default=0)
a = ap.parse_args(); cfg = json.load(open(a.cfg)); eng = C.Engine(a.model, a.data, a.out); eng.run_stats(1231)
fb = {}
for n in ("fb_annot", "fb_teach"):
    rows = [json.loads(l) for l in open(f"{a.data}/{n}.jsonl")]
    for r in rows: r["ids"] = eng.tok(eng.chat(r["user"]), add_special_tokens=False).input_ids
    fb[n] = rows
fbd = dict(ann=fb["fb_annot"], teach=fb["fb_teach"], old_labels=["Alarm", "Weather", "Music", "Calendar", "Lists", "News", "Transport"])
kept, by, nfb = eng.build(cfg); print("BYTES", json.dumps(by))
rq = dict(eb=cfg["eb"], eg=cfg.get("eg", 128)) if a.requant else None
res = FB.eval_feedback(eng, fbd, None, a.nlay, a.lam, a.margin, rq); res["cfg"] = cfg; res["bytes"] = by; res["args"] = vars(a)
json.dump(res, open(f"{a.out}/res/fb_{cfg['name']}_L{a.nlay}_lam{a.lam}_m{a.margin}_rq{a.requant}.json", "w"), indent=1); print("FBRESULT", json.dumps(res))
