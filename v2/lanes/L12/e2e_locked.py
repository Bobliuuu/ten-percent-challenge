"""(4) Locked exam scored from the PACKED loader (l12_runtime), no engine. --mode exam: stock packed model once (retention vs the stored base outputs of exam_outs_Q35_B3_rtn4_mass_k25.json, teach, neighbours).
--mode fb --draws 100,101,...: shipped apply_feedback (same code as feedback.py) per draw with real / shuffle / wrong labels; each scored on teach (164), neighbours (351) and the full exam; undo between runs.
The first real run additionally round-trips through fb.bin (save, undo, load_carrier) and checks teach outputs are identical."""
import sys, os, json, argparse, random, collections, hashlib, time
ap = argparse.ArgumentParser(); ap.add_argument("--mode", default="exam"); ap.add_argument("--draws", default="100"); ap.add_argument("--out"); ap.add_argument("--root", default="/workspace/fin"); ap.add_argument("--art", default="/r/art")
ap.add_argument("--exam_outs", default="exam_outs_Q35_B3_rtn4_mass_k25.json"); ap.add_argument("--nann", type=int, default=25); ap.add_argument("--nlay", type=int, default=10); ap.add_argument("--lam", type=float, default=2000.0); ap.add_argument("--awt", type=float, default=30.0); ap.add_argument("--m", type=float, default=64.0)
a = ap.parse_args(); ROOT = os.path.abspath(a.root); sys.path.insert(0, ROOT + "/v2/exam"); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch, loader as L, l12_runtime as RT
T0 = time.time(); log = lambda *x: print(f"[{time.time()-T0:6.0f}s]", *x, flush=True)
L.verify(); T = "SKILL7"; exam = L.load(T, "exam"); teach = L.load(T, "teach"); pool = L.load(T, "ann_pool"); neigh = set(L.load(T, "neighbours")); NEW = L.new_label(T); OLD = L.labels(T)
nb = [r for r in exam if r["id"] in neigh]; cal = [json.loads(l) for l in open(a.art + "/feedback_cal.jsonl")][:300]
EO = json.load(open(a.exam_outs)); assert EO["ids"] == [r["id"] for r in exam]; base = dict(zip(EO["ids"], EO["base"])); bok = {r["id"]: base[r["id"]] == r["answer"] for r in exam}
lm, tok = RT.load(a.art, carrier=False); R = RT.Runner(lm, tok); log("packed loaded")
for rows in (exam, teach, pool): R.encode(rows)
def evaluate():
    to = [o.strip() for o in R.gen(teach)]; eo = R.gen(exam); eok = {r["id"]: o == r["answer"] for r, o in zip(exam, eo)}
    nbr = L.report(nb, bok, eok); full = L.report(exam, bok, eok)
    return dict(teach_acc=sum(o == NEW for o in to) / len(teach), teach_top=collections.Counter(to).most_common(4), neigh_ret=nbr["overall"]["ret"], neigh_n=nbr["overall"]["n"], exam_ret=full["overall"]["ret"], exam_n=full["overall"]["n"], exam_wilson_lo=full["overall"]["wilson_lo"],
                exam_kind_min=min(v["ret"] for v in full["by_kind"].values() if v["ret"] is not None), by_kind={k: v["ret"] for k, v in full["by_kind"].items()}, new_label_leak_on_exam=sum(o == NEW for o in eo),
                agree_with_fakequant=sum(1 for i, o in zip([r["id"] for r in exam], eo) if o == dict(zip(EO["ids"], EO["quant"]))[i]) / len(exam), _teach_out=to, _exam_out=eo, _report=full)
res = dict(mode=a.mode, exam_manifest_sha=hashlib.sha256(open(ROOT + "/v2/exam/MANIFEST.json", "rb").read()).hexdigest())
if a.mode == "exam":
    r = evaluate(); json.dump(dict(ids=[x["id"] for x in exam], packed=r.pop("_exam_out"), teach=r.pop("_teach_out")), open(a.out + "/exam_outs_packed.json", "w")); rep = r.pop("_report"); res["empty"] = r; res["report"] = rep
    print("E2E EXAM", json.dumps(r)); json.dump(res, open(a.out + "/exam_final.json", "w"), indent=1); print("E2E DONE"); sys.exit(0)
mist = [r for r, o in zip(pool, R.gen(pool)) if o != NEW]; log("pool", len(pool), "mistakes", len(mist)); res["n_pool_mistakes"] = len(mist); res["runs"] = []
for d in [int(x) for x in a.draws.split(",")]:
    rnd = random.Random(d); m2 = mist[:]; rnd.shuffle(m2); ann = m2[:a.nann]; hsc = None
    for c in ("real", "shuffle", "wrong"):
        tg = [NEW] * len(ann) if c == "real" else [rnd.choice(OLD) for _ in ann] if c == "shuffle" else [OLD[-1]] * len(ann)
        rows = [dict(user=r["user"], answer=t, ids=r["ids"]) for r, t in zip(ann, tg)]
        if hsc is None: hsc = RT.cal_stats(R, cal, len(lm.model.layers) - a.nlay)
        carrier, undo = RT.apply_feedback(R, rows, cal, nlay=a.nlay, lam=a.lam, awt=a.awt, margin=a.m, hsc=hsc, log=lambda *x: None)
        cb = sum(v[1].numel() * 2 + v[0].numel() * 2 for v in carrier.values()); r = evaluate(); extra = {}
        if c == "real" and d == int(a.draws.split(",")[0]):
            RT.save_carrier("/tmp", carrier); undo(); assert RT.load_carrier(lm, "/tmp") == len(carrier); t2 = [o.strip() for o in R.gen(teach)]
            extra["file_roundtrip_teach_identical"] = (t2 == r["_teach_out"]); extra["fb_bin_bytes"] = os.path.getsize("/tmp/fb.bin"); RT.clear_carrier("/tmp")
        undo()
        r.pop("_teach_out"); r.pop("_exam_out"); r.pop("_report"); r.update(cond=c, draw=d, carrier_bytes=cb, n_ann=len(ann), **extra); res["runs"].append(r); print("E2E RUN", json.dumps(r), flush=True)
        json.dump(res, open(a.out + "/fb_partial.json", "w"), indent=1)
json.dump(res, open(a.out + "/fb_final.json", "w"), indent=1); print("E2E DONE")
