"""FINAL scoring (run ONCE per config) on the shared locked exam via v2/exam/loader.py.
python run_final.py --cfg best.json --task SKILL7 --out out_final [--root <repo root containing v2/exam>]
Steps: verify manifest; calibration = loader cal_big (SKILL7) / cal (BANK6); base and cfg real greedy generation on exam; report(); feedback (<=25 real mistakes from ann_pool,
real/shuffled/empty) on teach + neighbours + full exam; writes final_<name>.json."""
import sys, os, json, argparse, random, collections, hashlib
ap = argparse.ArgumentParser(); ap.add_argument("--cfg", required=True); ap.add_argument("--task", default="SKILL7"); ap.add_argument("--out", default="out_final"); ap.add_argument("--root", default="../../..")
ap.add_argument("--model", default="Qwen/Qwen3-30B-A3B-Instruct-2507"); ap.add_argument("--max_new", type=int, default=6); ap.add_argument("--nlay", type=int, default=4); ap.add_argument("--lam", type=float, default=1.0); ap.add_argument("--margin", type=float, default=2.0)
ap.add_argument("--requant", type=int, default=0); ap.add_argument("--nann", type=int, default=25); ap.add_argument("--no_fb", action="store_true")
a = ap.parse_args(); ROOT = os.path.abspath(a.root); sys.path.insert(0, ROOT + "/v2/exam"); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import loader as L, l12_core as C, l12_fb as FB
L.verify(); T = a.task; cfg = json.load(open(a.cfg)); os.makedirs(a.out + "/data", exist_ok=True)
w = lambda n, rows: open(f"{a.out}/data/{n}.jsonl", "w").write("".join(json.dumps(r) + "\n" for r in rows))
cal = L.load(T, "cal_big" if T == "SKILL7" else "cal"); sel = L.load(T, "sel"); w("cal", cal); w("sel", sel); w("rep", sel[:50])
exam = L.load(T, "exam"); teach = L.load(T, "teach"); pool = L.load(T, "ann_pool"); neigh = set(L.load(T, "neighbours")); NEW = L.new_label(T); OLD = L.labels(T)
eng = C.Engine(a.model, a.out + "/data", a.out); eng.max_new = a.max_new; eng.run_stats(len(cal))
tk = lambda rows: [dict(r, ids=eng.tok(eng.chat(r["user"]), add_special_tokens=False).input_ids) for r in rows]
exam, teach, pool = tk(exam), tk(teach), tk(pool)
res = dict(cfg=cfg, task=T, n_exam=len(exam), exam_manifest_sha=hashlib.sha256(open(ROOT + "/v2/exam/MANIFEST.json", "rb").read()).hexdigest())
eng.free_live(); eng.set_full_all(); eng.restore_nonexpert(); base = eng.gen_eval(exam)
ok = lambda outs: {r["id"]: o == r["answer"] for r, o in zip(exam, outs)}; bok = ok(base)
kept, by, nfb = eng.build(cfg); res["bytes"] = by; q = eng.gen_eval(exam); qok = ok(q)
res["report"] = L.report(exam, bok, qok); res["report_clean_only"] = L.report([r for r in exam if r.get("tier", "clean") == "clean"], bok, qok)
res["base_outs_top"] = collections.Counter(base).most_common(8); res["quant_flip_outs"] = collections.Counter(o for r, o, b in zip(exam, q, base) if bok[r["id"]] and not qok[r["id"]]).most_common(8)
json.dump(dict(base=base, quant=q, ids=[r["id"] for r in exam]), open(f"{a.out}/exam_outs_{cfg['name']}.json", "w"))
print("EXAM", json.dumps({k: res["report"][k] for k in ("overall", "gate_overall_98", "gate_kind_95", "base_acc")}), flush=True)
if not a.no_fb:
    rnd = random.Random(0); mist = [r for r in pool if eng.gen_eval([r])[0] != NEW]; rnd.shuffle(mist); ann = mist[:a.nann]; res["n_ann"] = len(ann); res["n_pool_mistakes"] = len(mist)
    nb = [r for r in exam if r["id"] in neigh]; fbres = {}
    rq = dict(eb=cfg["eb"], eg=cfg.get("eg", 128)) if a.requant and cfg.get("eb") else None
    for c in ("empty", "real", "shuffle"):
        if c == "empty": undo, info = (lambda: None), {}
        else:
            tg = [NEW] * len(ann) if c == "real" else [rnd.choice(OLD) for _ in ann]; undo, info = FB.fb_edit(eng, ann, tg, a.nlay, a.lam, a.margin, rq)
        to = [o.strip() for o in eng.gen_eval(teach)]; eo = eng.gen_eval(exam); eok = ok(eo)
        nbr = L.report(nb, bok, eok); full = L.report(exam, bok, eok)
        fbres[c] = dict(teach_acc=sum(o == NEW for o in to) / len(teach), teach_top=collections.Counter(to).most_common(4), neigh_ret=nbr["overall"], exam_ret=full["overall"], exam_kind_min=min(v["ret"] for v in full["by_kind"].values() if v["ret"] is not None),
                        new_label_leak_on_exam=sum(o == NEW for o in eo), info=info); undo(); print("FB", c, json.dumps(fbres[c]), flush=True)
    res["feedback"] = fbres; res["fb_args"] = dict(nlay=a.nlay, lam=a.lam, margin=a.margin, requant=a.requant)
json.dump(res, open(f"{a.out}/final_{cfg['name']}.json", "w"), indent=1); print("FINAL_DONE", cfg["name"])
