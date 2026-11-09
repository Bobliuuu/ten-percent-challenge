"""fb_eval <task> <mode dev|locked|proxy> <tag> '<json list of edit cfgs>' <ndraw>
Loads the PACKED artifact from /art/<task> (runtime.load), then for every edit cfg and draw: sample 12 new-label items from the annotation pool + 12 real mistakes of the compressed model on old items
(24 annotations), run the closed-form feedback edit (fbl3) under conditions real / shuffled (labels permuted among the annotation labels) / wrong (random wrong old label), evaluate teach + neighbours by REAL greedy generation, restore.
mode dev: pools = dev/ann split (NOT the locked exam): annotation candidates = first 60 of the ann pool (hash order), dev-teach = the other 90 (BANK6) / rest (SKILL7); neighbours = dev (+sel) items correct under empty feedback.
mode locked: annotations from the exam ann_pool, teach = locked teach set, neighbours = locked neighbour ids, keep-set = base-correct (base outputs file in /art/results). mode proxy: hostile own-data proxy (see proxy/)."""
import sys, os, json, random, hashlib, collections, math, time, shutil
import torch
import runtime, fbl3
tn, mode, tag, CFGS, NDRAW = sys.argv[1], sys.argv[2], sys.argv[3], json.loads(sys.argv[4]), int(sys.argv[5])
ART = f"/art/{tn}"; T0 = time.time(); log = lambda *a: print(f"[{time.time()-T0:6.0f}s]", *a, flush=True)
for f in ("task.json", "calib_prompts.jsonl"):
    if not os.path.exists(f"{ART}/{f}"): shutil.copy(f"/root/l3/pkg_{tn}/{f}", f"{ART}/{f}")
for f in os.listdir(ART):
    if f.startswith("edit_"): os.remove(f"{ART}/{f}")
model, tok, task = runtime.load(ART); tok.padding_side = "left"; log("loaded"); dev = "cuda"
PXF = sys.argv[6] if len(sys.argv) > 6 else None
NEW = PXF.split(":")[1] if PXF else {"skill7x": "Recipe", "bank6x": "Security"}[tn]; PFX = task["prompt"].split("{X}")[0]
rd = lambda p: [json.loads(l) for l in open(p)]
H = lambda s: int(hashlib.sha256(s.encode()).hexdigest(), 16)
def txt(r): u = r["user"]; assert u.startswith(PFX), u[:80]; return u[len(PFX):]
D = "/root/data/"; E = "/root/exam/data/"; ds = tn
cal_pk = rd(f"{ART}/calib_prompts.jsonl"); calset = {r["text"] for r in cal_pk}
if mode == "dev":
    pool = sorted(rd(D + f"{ds}_ann.jsonl"), key=lambda r: H("fbd|" + r["id"])); cand, teach = pool[:60], pool[60:]
    neigh = rd(D + f"{ds}_dev.jsonl") + rd(D + f"{ds}_sel.jsonl")[::(1 if tn == "bank6x" else 3)]; keep_ids = None
    errpool = [r for r in rd(D + f"{ds}_cal.jsonl") if txt(r) not in calset][:600]
elif mode == "locked":
    sp = {"skill7x": "skill7", "bank6x": "bank6"}[tn]; cand = rd(E + f"{sp}_ann_pool.jsonl"); teach = rd(E + f"{sp}_teach.jsonl"); nid = set(json.load(open(E + f"{sp}_neighbours.json")))
    neigh = [r for r in rd(E + f"{sp}_exam.jsonl") if r["id"] in nid]; bo = json.load(open(f"/art/results/base_{tn}_EXAM_1.json")); bo = dict(zip(bo["ids"], bo["outs"]))
    keep_ids = {r["id"] for r in neigh if bo[r["id"]] == r["answer"]}
    errpool = [r for r in rd(D + f"{ds}_cal.jsonl") if txt(r) not in calset][:600]
elif mode == "proxy":
    P = "/root/l3/proxy/"; oldf = "old_skills.jsonl" if tn == "skill7x" else "bank_old.jsonl"; mk = lambda r: dict(id=r["id"], user=PFX + r["text"], answer=r["kind"], src="px", kind=r["kind"] + "/" + r["style"])
    oldr = [mk(r) for r in rd(P + oldf)]; newr = sorted([mk(r) | dict(src=r["style"]) for r in rd(P + PXF.split(":")[0])], key=lambda r: H("pxa|" + r["id"])); cand, teach = newr[:40], newr[40:]
    neigh = [r for r in oldr if H("pxc|" + r["id"]) % 2 == 1]; errpool = [r for r in oldr if H("pxc|" + r["id"]) % 2 == 0]; keep_ids = None
    cal_pk = rd(f"{ART}/calib_prompts.jsonl")
@torch.no_grad()
def gen(texts, bs=96, max_new=6):
    ps = [tok.apply_chat_template([{"role": "user", "content": task["prompt"].replace("{X}", t)}], tokenize=False, add_generation_prompt=True) for t in texts]
    order = sorted(range(len(ps)), key=lambda i: len(ps[i])); out = [None] * len(ps)
    for s0 in range(0, len(order), bs):
        ix = order[s0:s0 + bs]; enc = tok([ps[i] for i in ix], return_tensors="pt", padding=True, add_special_tokens=False).to(dev)
        g = model.generate(**enc, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.pad_token_id)
        for i, o in zip(ix, tok.batch_decode(g[:, enc.input_ids.shape[1]:], skip_special_tokens=True)): out[i] = o.strip()
    return out
LAYALL = sorted({l for c in CFGS for l in c["layers"]})
fb = fbl3.FB(model, tok, task, cal_pk, LAYALL, dev); log("cov done", len(cal_pk), "calib prompts, layers", LAYALL[0], LAYALL[-1])
ntx = [txt(r) for r in neigh]; eo = gen(ntx); ok0 = [o == r["answer"] for o, r in zip(eo, neigh)]
if keep_ids is None: keep = [i for i, r in enumerate(neigh) if ok0[i]]
else: keep = [i for i, r in enumerate(neigh) if r["id"] in keep_ids]
if mode == "proxy": log("proxy empty acc on all old-proxy neigh", sum(ok0) / len(ok0))
log("neighbours", len(neigh), "keep", len(keep), "empty acc on keep", sum(ok0[i] for i in keep) / len(keep))
teach_t = [txt(r) for r in teach]; to0 = gen(teach_t); log("empty teach", sum(o == NEW for o in to0) / len(teach))
er = errpool; eo2 = gen([txt(r) for r in er]); old_err = [dict(text=txt(r), label=r["answer"]) for r, o in zip(er, eo2) if o != r["answer"]]; log("old-skill errors of the packed model on the error pool:", len(old_err), "of", len(er))
cells = lambda idx, outs: {k: v for k, v in sorted(((c, [0, 0]) for c in []))}
def evaluate(cond, full):
    ix = keep if full else keep[::3]; no = gen([ntx[i] for i in ix]); to = gen(teach_t)
    ok = [o == neigh[i]["answer"] for o, i in zip(no, ix)]; cl = collections.defaultdict(lambda: [0, 0])
    for o, i in zip(no, ix): c = cl[neigh[i]["src"] + "|" + neigh[i]["kind"]]; c[0] += 1; c[1] += int(o == neigh[i]["answer"])
    tb = collections.defaultdict(lambda: [0, 0])
    for o, r in zip(to, teach): c = tb[r["src"]]; c[0] += 1; c[1] += int(o == NEW)
    return dict(teach=sum(o == NEW for o in to) / len(to), teach_by_src={k: round(v[1] / v[0], 3) for k, v in tb.items()}, neigh=sum(ok) / len(ok), n_neigh=len(ok), lost=len(ok) - sum(ok),
                min_cell=min([v[1] / v[0] for v in cl.values() if v[0] >= 10] or [1.0]), leak=sum(o == NEW for o in no) / len(no), cells={k: round(v[1] / v[0], 4) for k, v in sorted(cl.items())},
                teach_outs=dict(collections.Counter(to).most_common(6)))
RES = dict(task=tn, mode=mode, tag=tag, n_keep=len(keep), n_teach=len(teach), n_old_err=len(old_err), empty=dict(teach=sum(o == NEW for o in to0) / len(teach), neigh=sum(ok0[i] for i in keep) / len(keep)), runs=[])
outp = f"/art/results/fb_{mode}_{tn}_{tag}.json"; RES["pxf"] = PXF
for ci, cfg in enumerate(CFGS):
    ec = {k: v for k, v in cfg.items() if k not in ("name", "conds", "n_new", "n_old")}; conds = cfg.get("conds", ["real", "shuffled", "wrong"])
    for d in range(NDRAW):
        rng = random.Random(100 + d); nn = cfg.get("n_new", 12); no_ = cfg.get("n_old", 12); an = rng.sample(cand, nn); ao = rng.sample(old_err, min(no_, len(old_err)))
        new_rows = [dict(text=txt(r), label=NEW) for r in an]; rows = new_rows + ao; skills = sorted({r["answer"] for r in neigh})
        for cond in conds:
            if cond == "real": its = rows
            elif cond == "shuffled": olds = [r["label"] for r in ao]; its = [dict(text=r["text"], label=olds[i % len(olds)]) for i, r in enumerate(new_rows)] + [dict(text=r["text"], label=NEW) for r in ao]
            elif cond == "wrong": its = [dict(text=r["text"], label=rng.choice([s for s in skills if s != r["label"]])) for r in rows]
            t1 = time.time(); new = fb.edit(its, verbose=False, **ec); by = fbl3.edit_bytes(new); res = evaluate(cond, cond == "real"); fb.restore()
            res.update(cfg=cfg.get("name", ""), draw=d, cond=cond, edit_bytes=by, secs=round(time.time() - t1)); RES["runs"].append(res)
            log(cfg.get("name", ci), "draw", d, cond, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in res.items() if k not in ("cells", "cfg", "teach_outs")})
    json.dump(RES, open(outp, "w"))
json.dump(RES, open(outp, "w")); agg = collections.defaultdict(list)
for r in RES["runs"]: agg[(r["cfg"], r["cond"])].append(r)
for (c, cond), v in agg.items(): print("SUMMARY", tn, mode, c, cond, "n", len(v), "teach %.3f" % (sum(x["teach"] for x in v) / len(v)), "neigh %.4f" % (sum(x["neigh"] for x in v) / len(v)), "min_cell %.3f" % min(x["min_cell"] for x in v), "leak %.4f" % (sum(x["leak"] for x in v) / len(v)), "MB %.1f" % (v[0]["edit_bytes"] / 1e6), flush=True)
print("RESULT_JSON", json.dumps(RES), flush=True)
