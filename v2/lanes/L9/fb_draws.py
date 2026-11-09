"""Feedback draws on the PACKED artifact (L11 recipe: o_proj, last 8 layers, protected closed form). usage: python fb_draws.py ART OUT dev|locked [seeds comma] [lam] [margin]
dev    : config pick on ann_pool/dev ONLY. annotate from ann_pool[:150] (12 Recipe) + 13 real old-skill mistakes on cal; dev-teach = ann_pool[150:] (150 unseen Recipe items, never annotated); dev neighbours = base-correct items of dev SEL (skill7_sel).
locked : fixed config; per seed real / shuffled (labels permuted among the 25 annotation labels) / wrong; plus empty once. teach = locked shared teach (164), neighbours = base-correct of the 379 shared neighbour ids of the exam.
Real greedy generation, exact label match."""
import sys, os, json, random, collections, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, HERE + "/../../exam")
import torch, packed_loader as PL, feedback as FB
ART, OUT, PHASE = sys.argv[1:4]; SEEDS = [int(x) for x in (sys.argv[4] if len(sys.argv) > 4 else "0,1").split(",")]
NEWL = "Recipe"; SK = ["Alarm", "Weather", "Music", "Calendar", "Lists", "News", "Transport"]
import loader as EX
cal = [json.loads(l) for l in open(os.path.join(ART, "feedback_cal.jsonl"))]; pool = EX.load("SKILL7", "ann_pool")
t0 = time.time(); m, tok = PL.build(ART); print("loaded", round(time.time() - t0), flush=True)
cov_t = time.time()
def gen(rows): return PL.generate(m, tok, [r["user"] for r in rows])
if PHASE == "dev":
    sel = [json.loads(l) for l in open(HERE + "/data/skill7_sel.jsonl")]; bo = json.load(open(HERE + "/results/outs_skill7_base_SEL.json"))["outs"]
    nbr = [r for r, o in zip(sel, bo) if o == r["answer"]][:500]; teach = pool[150:]; annpool = pool[:150]
else:
    teach = EX.load("SKILL7", "teach"); NB = set(EX.load("SKILL7", "neighbours")); ex = {r["id"]: r for r in EX.load("SKILL7", "exam")}
    b = json.load(open(HERE + "/results/base_skill7x_SEL_1.json")); bo = dict(zip(b["ids"], b["outs"])); nbr = [ex[i] for i in sorted(NB) if i in ex and bo.get(i) == ex[i]["answer"]]; annpool = pool
print("teach", len(teach), "neigh", len(nbr), flush=True)
co = gen(cal); ERR = [r for r, o in zip(cal, co) if o != r["answer"]]; print("old-skill errors on cal", len(ERR), flush=True)
X = FB.Ctx(m, tok, cal); print("cov done", round(time.time() - cov_t), flush=True)
def annot(seed):
    rg = random.Random(100 + seed); return rg.sample(annpool, 12) + rg.sample(ERR, min(13, len(ERR)))
def ev(tag, extra=None):
    t = gen(teach); n = gen(nbr); cells = collections.defaultdict(lambda: [0, 0])
    for o, r in zip(n, nbr): c = cells[r["src"] + "|" + r["kind"]]; c[0] += 1; c[1] += int(o == r["answer"])
    d = dict(tag=tag, teach=sum(o == NEWL for o in t) / len(t), neigh=sum(o == r["answer"] for o, r in zip(n, nbr)) / len(nbr), n_neigh=len(nbr), min_cell=min(v[1] / v[0] for v in cells.values()),
             leak=sum(o == NEWL for o in n) / len(n), teach_outs=dict(collections.Counter(t).most_common(5)), **(extra or {}))
    print("RESULT_JSON", json.dumps(d), flush=True); return d
res = []
def one(mode, seed, lam, margin):
    a = annot(seed)
    if mode == "wrong": rg = random.Random(seed); a = [dict(r, answer=rg.choice([x for x in SK + [NEWL] if x != r["answer"]])) for r in a]
    car = FB.make_carrier(X, a, "real" if mode == "wrong" else mode, seed, lam=lam, margin=margin)
    d = ev(f"{mode}{seed}_lam{lam:g}_m{margin:g}", dict(mode=mode, seed=seed, lam=lam, margin=margin, n_ann=len(a), n_ann_new=sum(r["answer"] == NEWL for r in a)))
    if mode == "real" and PHASE == "locked":
        from safetensors.torch import save_file; save_file(car, f"{OUT}.carrier_seed{seed}.safetensors")
    FB.undo(X, car); res.append(d)
if PHASE == "dev":
    res.append(ev("empty"))
    for lam, mg in [(5000, 32), (5000, 16), (1000, 32), (20000, 32), (5000, 64)]:
        for s in SEEDS:
            one("real", s, lam, mg)
        one("shuffled", SEEDS[0], lam, mg)
else:
    lam = float(sys.argv[5]); mg = float(sys.argv[6])
    if 0 in SEEDS: res.append(ev("empty"))
    for s in SEEDS:
        for mode in ("real", "shuffled", "wrong"): one(mode, s, lam, mg)
json.dump(res, open(OUT, "w"), indent=1); print("DONE", round(time.time() - t0))
