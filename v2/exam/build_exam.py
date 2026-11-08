"""Build the SHARED locked exam files (SKILL7 + BANK6) + teach / annotation / neighbour sets. Deterministic. Public data only.
Run BEFORE any model sees the files: python3 -I build_exam.py  -> data/*.jsonl, MANIFEST.json (sha256 of every file).
Policy (see EXAM_PLAN.md): an item is eligible only if its normalised text does not occur in any non-locked repo data file (dev/calib/sel/feedback/bundles).
tier 'clean' = occurs nowhere; tier 'seen_tox'/'seen_l5dev' = occurs only in a measurement-only dev file of another task (flagged per item, used only to fill kinds that would otherwise fall under 300).
No locked/sealed file of any lane is read (leakscan skips exam|lock|seal|x15|never paths)."""
import json, os, sys, hashlib, collections, re, random
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import pools as P, leakscan as L
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
OUT = os.path.join(HERE, "data"); os.makedirs(OUT, exist_ok=True)
H = P.H; norm = P.norm
PLAN = json.load(open(f"{ROOT}/v2/task_search/EASY/PLAN.json")); PLAN2 = json.load(open(f"{ROOT}/v2/task_search/EASY/PLAN2.json"))
PROMPT = {"SKILL7": PLAN["SKILL7"]["prompt"], "BANK6": PLAN2["BANK6"]["prompt"]}      # identical to the lanes' current prompts
TL = 900                                                                                  # same truncation as build_easy (tl*3)
# ---- which files saw which texts
def group(p):
    if re.search(r"tournament/TOX/routing/data/(dev|train|probe_dev)\.jsonl", p): return "tox"
    if p.endswith("lanes/L5/data/ct_items.jsonl"): return "l5dev"
    return "used"
def seen_map(cands):
    from multiprocessing import Pool
    fl = [p for p in L.files(ROOT)]
    seen = collections.defaultdict(set)
    with Pool(6) as pl:
        for p, s in zip(fl, pl.imap(L.scan, fl, chunksize=4)):
            g = group(p)
            for k in s & cands: seen[k].add(g)
    return seen
def item(task, role, r, extra=None):
    d = dict(id=f"{task}|{r['src']}|{r['rid']}", task=task, role=role, src=r["src"], kind=r["label"], user=PROMPT[task].replace("{X}", P.trunc(r["text"], TL)), answer=r["label"], tier=r["tier"])
    return d
def order(task, r): return H(f"exam-v1|{task}|{r['src']}|{r['rid']}")
def dedupe(R, task):
    seen = set(); U = []
    for r in R:
        k = norm(r["text"])
        if not k or k in seen: continue
        seen.add(k); r["k"] = k; r["b"] = P.bucket(task, r["text"]); U.append(r)
    return U
def pick(R, n, task, tiers=("clean",)):
    """clean first (hash order), then seen_* ; n items"""
    out = []
    for t in ("clean", "seen_tox", "seen_l5dev"):
        if t not in tiers: continue
        out += sorted([r for r in R if r["tier"] == t], key=lambda r: order(task, r))
    return out[:n]
def write(name, rows):
    p = f"{OUT}/{name}.jsonl"; open(p, "w").write("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows)); return p
def main():
    s7 = dedupe(P.s7_pools(), "SKILL7"); b6 = dedupe(P.b6_pools(), "BANK6")
    cands = {r["k"] for r in s7 + b6}; seen = seen_map(cands)
    for r in s7 + b6:
        g = seen.get(r["k"], set())
        r["tier"] = "clean" if not g else ("seen_tox" if g == {"tox"} else "seen_l5dev" if g == {"l5dev"} else "used")
    # ================= SKILL7
    T = lambda r: r["src"] in ("massive", "clinc150")
    by = lambda R, lab, f: [r for r in R if r["label"] == lab and f(r)]
    Q7 = {"Alarm": (131, 89, 200), "Weather": (141, 37, 242), "Music": (176, 100, 144), "Calendar": (260, 100, 0), "Lists": (232, 112, 0), "News": (91, 64, 265), "Transport": (224, 136, 0)}  # (T cap, hwu cap, mtop cap)
    exam7 = []
    for lab, (tc, hc, mc) in Q7.items():
        t = pick([r for r in by(s7, lab, T) if r["b"] >= 70], tc, "SKILL7", ("clean", "seen_tox"))
        h = pick(by(s7, lab, lambda r: r["src"] == "hwu64"), hc, "SKILL7")
        m = pick(by(s7, lab, lambda r: r["src"] == "mtop"), mc, "SKILL7")
        exam7 += [item("SKILL7", "exam", r) for r in t + h + m]
    exam7.sort(key=lambda d: H("e7" + d["id"]))
    ex7 = {norm(d["user"].split("Request: ", 1)[1].removesuffix(" ...")) for d in exam7}
    # teach (new skill Recipe): same-source clean T >=70 + cross-source never-used MTOP/HWU
    NP = PROMPT["SKILL7"]
    rec = [r for r in s7 if r["label"] == "Recipe"]
    teach7 = pick([r for r in rec if T(r) and r["b"] >= 70], 115, "SKILL7") + pick([r for r in rec if r["src"] == "mtop"], 100, "SKILL7") + pick([r for r in rec if r["src"] == "hwu64"], 15, "SKILL7")
    teach7 = [item("SKILL7", "teach", r) for r in teach7]
    ann_src = sorted([r for r in rec if T(r) and r["b"] < 70], key=lambda r: order("SKILL7", r))
    ann7 = [item("SKILL7", "ann_pool", r) for r in ann_src[:300]]
    FOOD = re.compile(r"\b(cook|recipe|food|dinner|lunch|breakfast|bake|baking|ingredient|grocer|grocery|shopping|meal|restaurant|eat|kitchen|oven|bread|cake|pizza|soup|dish|menu|buy)", re.I)
    near7 = sorted(d["id"] for d in exam7 if d["kind"] == "Lists" or FOOD.search(d["user"].split("Request: ", 1)[1]))
    # ================= BANK6
    Bk = lambda r: r["src"] == "banking77"
    Q6 = {"Card": (200, 200), "Transfer": (200, 200), "Cash": (200, 200), "TopUp": (360, 0), "Identity": (330, 0), "Exchange": (400, 0)}  # (banking77 cap, bitext cap)
    exam6 = []
    for lab, (bc, tc) in Q6.items():
        b = pick(by(b6, lab, Bk), bc, "BANK6", ("clean", "seen_l5dev"))
        x = pick(by(b6, lab, lambda r: r["src"] == "bitext_bank"), tc, "BANK6")
        exam6 += [item("BANK6", "exam", r) for r in b + x]
    exam6.sort(key=lambda d: H("e6" + d["id"]))
    ex6ids = {d["id"] for d in exam6}
    sec = [r for r in b6 if r["label"] == "Security"]
    tb = pick([r for r in sec if Bk(r)], 100, "BANK6") + pick([r for r in sec if r["src"] == "bitext_bank"], 100, "BANK6")
    teach6 = [item("BANK6", "teach", r) for r in tb]
    tids = {d["id"] for d in teach6}
    annp = sorted([r for r in sec if Bk(r) and f"BANK6|banking77|{r['rid']}" not in tids and r["tier"] != "used"] + [r for r in sec if Bk(r) and r["tier"] == "used" and f"BANK6|banking77|{r['rid']}" not in tids], key=lambda r: order("BANK6", r))
    ann6 = [item("BANK6", "ann_pool", r) for r in annp[:150]]
    SEC = re.compile(r"(lost|stolen|steal|pin|password|passcode|fraud|hack|secur|block|compromis|phone|suspicious|someone|unauthori)", re.I)
    near6 = sorted(d["id"] for d in exam6 if d["kind"] in ("Card", "Identity") or SEC.search(d["user"].split("Message: ", 1)[1]))
    dev6 = {norm(json.loads(l)["user"].split("Message: ", 1)[1].removesuffix(" ...")) for l in open(f"{ROOT}/v2/task_search/EASY/items2_dev.jsonl") if json.loads(l)["task"] == "BANK6"}
    taken = {r["k"] for r in b6 if f"BANK6|{r['src']}|{r['rid']}" in ex6ids or f"BANK6|{r['src']}|{r['rid']}" in tids} | {r["k"] for r in annp[:150]}
    rest = [r for r in b6 if Bk(r) and r["label"] != "Security" and r["k"] not in taken and r["k"] not in dev6]
    cal6 = []; sel6 = []
    for lab in Q6:
        cal6 += [item("BANK6", "cal", r) for r in sorted([r for r in rest if r["label"] == lab and r["b"] < 40], key=lambda r: order("BANK6", r))[:120]]
        sel6 += [item("BANK6", "sel", r) for r in sorted([r for r in rest if r["label"] == lab and 40 <= r["b"] < 70], key=lambda r: order("BANK6", r))[:60]]
    # ---- sanity: disjointness inside and against existing dev/cal/sel files
    for nm, rows in (("exam7", exam7), ("teach7", teach7), ("ann7", ann7), ("exam6", exam6), ("teach6", teach6), ("ann6", ann6), ("cal6", cal6), ("sel6", sel6)):
        assert len({d["id"] for d in rows}) == len(rows), nm
    txt = lambda d: norm(d["user"].split(": ", 1)[-1].split("\n")[-1].split("Request: ")[-1].split("Message: ")[-1].removesuffix(" ..."))
    for a, b in ((exam7, teach7), (exam7, ann7), (teach7, ann7), (exam6, teach6), (exam6, ann6), (teach6, ann6), (exam6, cal6), (exam6, sel6), (cal6, sel6)):
        sa = {txt(d) for d in a}; assert not any(txt(d) in sa for d in b), "overlap"
    files = {"skill7_exam": exam7, "skill7_teach": teach7, "skill7_ann_pool": ann7, "bank6_exam": exam6, "bank6_teach": teach6, "bank6_ann_pool": ann6, "bank6_cal": cal6, "bank6_sel": sel6}
    man = {"version": 1, "builder": "v2/exam/build_exam.py", "pools": "v2/exam/pools.py", "files": {}, "neighbours": {}}
    for nm, rows in files.items():
        p = write(nm, rows); man["files"][nm] = dict(path=os.path.relpath(p, ROOT), sha256=hashlib.sha256(open(p, "rb").read()).hexdigest(), n=len(rows),
                                                      by_kind=dict(collections.Counter(d["kind"] for d in rows)), by_src=dict(collections.Counter(d["src"] for d in rows)), by_tier=dict(collections.Counter(d["tier"] for d in rows)))
    for nm, ids in (("skill7_neighbours", near7), ("bank6_neighbours", near6)):
        p = f"{OUT}/{nm}.json"; json.dump(ids, open(p, "w"), indent=0); man["neighbours"][nm] = dict(path=os.path.relpath(p, ROOT), sha256=hashlib.sha256(open(p, "rb").read()).hexdigest(), n=len(ids))
    man["builder_sha256"] = hashlib.sha256(open(__file__, "rb").read()).hexdigest()
    man["cells"] = {nm: {f"{k}|{s}": c for (k, s), c in sorted(collections.Counter((d["kind"], d["src"]) for d in rows).items())} for nm, rows in files.items()}
    json.dump(man, open(f"{HERE}/MANIFEST.json", "w"), indent=1, sort_keys=True)
    for nm, f in man["files"].items(): print(nm, f["n"], f["by_kind"], f["by_src"], f["by_tier"])
    print("neighbours", {k: v["n"] for k, v in man["neighbours"].items()})
if __name__ == "__main__": main()
