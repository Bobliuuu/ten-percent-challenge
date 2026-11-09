"""L4 evaluation harness (Modal, NOT shipped). Stages: exam (hash check vs in-process + locked exam retention, once per packed body), fbdev (dev-proxy feedback tuning), fblock (locked teach + neighbours, real/shuffled/wrong/empty, >=6 draws)."""
import os, sys, re, json, time, math, random, hashlib, collections
sys.path.insert(0, "/root/l4"); sys.path.insert(0, "/root")
import torch
import l4fmt as F4, l4fb
T0 = time.time(); log = lambda *a: print(f"[{time.time()-T0:6.0f}s]", *a, flush=True)
NORM = lambda t: re.sub(r"^\s*category\s*:\s*", "", (t or "").strip(), flags=re.I).strip()
NEW = {"SKILL7": "Recipe", "BANK6": "Security"}; DS = {"SKILL7": "s7", "BANK6": "b6"}
NEVER = {"SKILL7": {"hwu64", "mtop"}, "BANK6": {"bitext_bank"}}
def wilson(k, n, z=1.96):
    if n == 0: return 0.0
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); r = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)); return (c - r) / d

class Ev:
    def __init__(s, lm, tok, task):
        s.lm, s.tok, s.task, s.ds = lm, tok, task, DS[task]; s.cache = {}
        s.chat = lambda u: tok.apply_chat_template([{"role": "user", "content": u}], tokenize=False, add_generation_prompt=True)
    def rows(s, n):
        if n not in s.cache:
            rs = [json.loads(l) for l in open(f"/root/data/{s.ds}_{n}.jsonl")]
            for r in rs: r["ids"] = s.tok(s.chat(r["user"]), add_special_tokens=False).input_ids
            s.cache[n] = rs
        return s.cache[n]
    def mk(s, rows):
        for r in rows:
            if "ids" not in r: r["ids"] = s.tok(s.chat(r["user"]), add_special_tokens=False).input_ids
        return rows
    @torch.no_grad()
    def gen(s, rows, budget=40000, max_new=5):
        outs = [None] * len(rows); tok = s.tok
        order = sorted(range(len(rows)), key=lambda i: len(rows[i]["ids"])); i = 0
        while i < len(order):
            j = i
            while j < len(order) and (j - i + 1) * len(rows[order[j]]["ids"]) <= budget and (j - i) < 128: j += 1
            j = max(j, i + 1); idx = order[i:j]; i = j
            L = max(len(rows[k]["ids"]) for k in idx); ids = torch.full((len(idx), L), tok.pad_token_id); am = torch.zeros((len(idx), L), dtype=torch.long)
            for r, k in enumerate(idx): ids[r, L - len(rows[k]["ids"]):] = torch.tensor(rows[k]["ids"]); am[r, L - len(rows[k]["ids"]):] = 1
            g = s.lm.generate(input_ids=ids.cuda(), attention_mask=am.cuda(), max_new_tokens=max_new, do_sample=False, pad_token_id=tok.pad_token_id)
            for k, t in zip(idx, tok.batch_decode(g[:, L:], skip_special_tokens=True)): outs[k] = NORM(t)
        return outs

def load_model(body, log=log):
    hashes = {}; lm, tok, rd = F4.load_packed(body, hashes=hashes, log=log); return lm, tok, rd, hashes

def stage_exam(task, body, cfg, tag, art):
    ds = DS[task]; lm, tok, rd, hashes = load_model(body); ev = Ev(lm, tok, task)
    ref = json.load(open(os.path.dirname(body) + "/hashes.json")); bad = [k for k in ref if hashes.get(k) != ref[k]]
    log("hash check: n", len(ref), "mismatch", len(bad), bad[:5])
    res = dict(task=task, tag=tag, hash_n=len(ref), hash_mismatch=len(bad), hash_bad_examples=bad[:10], bytes=os.path.getsize(body))
    exam = ev.rows("exam"); outs = ev.gen(exam); base = json.load(open(f"/r/res/base_{ds}_exam_1.json")); assert base["ids"] == [r["id"] for r in exam]
    bc = [NORM(o) == r["answer"] for o, r in zip(base["outs"], exam)]; ok = [o == r["answer"] for o, r in zip(outs, exam)]
    def ret(sel):
        n = sum(1 for i in sel if bc[i]); k = sum(1 for i in sel if bc[i] and ok[i]); return dict(n_bc=n, kept=k, ret=k / max(n, 1), lo=wilson(k, n))
    allx = list(range(len(exam))); res["overall"] = ret(allx); res["base_acc"] = sum(bc) / len(bc); res["acc"] = sum(ok) / len(ok)
    res["kinds"] = {k: ret([i for i, r in enumerate(exam) if r["kind"] == k]) for k in sorted({r["kind"] for r in exam})}
    res["src"] = {k: ret([i for i, r in enumerate(exam) if r["src"] == k]) for k in sorted({r["src"] for r in exam})}
    res["never_used"] = ret([i for i, r in enumerate(exam) if r["src"] in NEVER[task]]); res["clean_only"] = ret([i for i, r in enumerate(exam) if not str(r.get("tier", "")).startswith("seen")])
    res["n_exam"] = len(exam); res["min_kind"] = min(v["ret"] for v in res["kinds"].values())
    ids = [r["id"] for r in exam]; json.dump(dict(ids=ids, outs=outs), open(f"/a/{task}/exam_outs_{tag}.json", "w")); json.dump(res, open(f"/a/{task}/exam_{tag}.json", "w")); art.commit()
    print("RESULT_JSON", json.dumps(res), flush=True); return res

def build_cal(ev, n):
    cal = ev.rows("cal"); rng = random.Random(1); cal = rng.sample(cal, min(n, len(cal))); return [(r["ids"], r["answer"]) for r in cal]

def dev_proxy(ev, task):
    """dev proxy for a new label: hold out one menu class of the dev/sel split, relabel as a new word; prompts of that class drop it from the menu (as the locked new label is not in the menu)."""
    sel = ev.rows("sel")
    if task == "SKILL7": hold, new, menu = "Music", "Playlist", "Music, "
    else: hold, new, menu = "Exchange", "Forex", ", Exchange"
    pool_all = [r for r in sel if r["kind"] == hold]; pool_all.sort(key=lambda r: hashlib.md5(r["id"].encode()).hexdigest())
    npool = 60 if task == "SKILL7" else 14; pool, teach = pool_all[:npool], pool_all[npool:]
    for r in pool_all:
        r["user"] = r["user"].replace(menu, "", 1); r["ids"] = ev.tok(ev.chat(r["user"]), add_special_tokens=False).input_ids
    others = [r for r in sel if r["kind"] != hold]; rng = random.Random(5); neigh = rng.sample(others, 450)
    return new, pool, teach, neigh

def run_draws(ev, fb, task, new, pool, teach, neigh, old_err, cfg, draws, conds, n_new=14, n_old=10, label=""):
    """returns dict of per-draw per-condition results. neigh must already be base(=packed pre-edit)-correct"""
    labels_all = sorted({r["answer"] for r in ev.rows("cal")})
    def evaluate(tag):
        to = ev.gen(teach); ta = [o == new for o in to]; cells = collections.defaultdict(list)
        for r, a in zip(teach, ta): cells[r.get("src", "?")].append(a)
        no = ev.gen(neigh); ok = [o == r["answer"] for o, r in zip(no, neigh)]; nc = collections.defaultdict(lambda: [0, 0])
        for o, r in zip(no, neigh): c = nc[r["src"] + "|" + r["kind"]]; c[0] += 1; c[1] += int(o == r["answer"])
        plural = sum(1 for o in to if o.lower().startswith(new.lower()) and o != new)
        return dict(tag=tag, teach=sum(ta) / len(ta), teach_lo=wilson(sum(ta), len(ta)), teach_src={k: dict(n=len(v), rate=sum(v) / len(v)) for k, v in sorted(cells.items())},
                    teach_variant_of_new=plural, teach_samples=collections.Counter(to).most_common(5), neigh=sum(ok) / len(ok), neigh_lo=wilson(sum(ok), len(ok)), n_neigh=len(ok),
                    neigh_min_cell=min(v[1] / v[0] for v in nc.values()), neigh_cells={k: [v[1], v[0]] for k, v in sorted(nc.items())}, leak=sum(1 for o in no if o == new) / len(no))
    out = dict(cfg=cfg, label=label, draws=[]); fb.restore()
    if "empty" in conds: out["empty"] = evaluate("empty"); log("empty", {k: v for k, v in out["empty"].items() if k not in ("neigh_cells",)})
    srcs = sorted({r["src"] for r in pool})
    for d in range(draws):
        rng = random.Random(100 + d); by = collections.defaultdict(list)
        for r in pool: by[r["src"]].append(r)
        for v in by.values(): rng.shuffle(v)
        ann_new = []; i = 0                          # stratified over source cells (round robin)
        while len(ann_new) < n_new:
            sname = srcs[i % len(srcs)]; i += 1
            if by[sname]: ann_new.append(by[sname].pop())
        ann_old = rng.sample(old_err, min(n_old, len(old_err))); items = [(r["ids"], new) for r in ann_new] + list(ann_old); labs = [l for _, l in items]
        dr = dict(draw=d, n_ann=len(items), ann_src=collections.Counter(r["src"] for r in ann_new))
        for cond in conds:
            if cond == "empty": continue
            if cond == "real": its = items
            elif cond == "shuffled":
                for _ in range(50):
                    l2 = labs[:]; rng.shuffle(l2)
                    if l2 != labs: break
                its = [(p, l) for (p, _), l in zip(items, l2)]
            elif cond == "wrong": its = [(p, rng.choice([x for x in labels_all if x != l])) for p, l in items]
            ns = fb.edit(its); res = evaluate(cond); res["n_sites"] = ns; dr[cond] = res; fb.restore()
            log("draw", d, cond, {k: v for k, v in res.items() if k not in ("neigh_cells", "teach_src")})
        out["draws"].append(dr)
    return out

def stage_fb(task, body, cfg, tag, art, dev):
    lm, tok, rd, hashes = load_model(body); ev = Ev(lm, tok, task); cal = build_cal(ev, cfg.get("ncovs", 300))
    cfgs = cfg["cfgs"]; draws = cfg.get("draws", 6); conds = cfg.get("conds", ["empty", "real", "shuffled", "wrong"])
    errs = ev.rows("cal")[:cfg.get("nerr_pool", 1500)]; eo = ev.gen(errs); old_err = [(r["ids"], r["answer"]) for r, o in zip(errs, eo) if o != r["answer"]]; log("old errs", len(old_err), "of", len(errs))
    if dev: new, pool, teach, neigh = dev_proxy(ev, task); no = ev.gen(neigh); neigh = [r for r, o in zip(neigh, no) if o == r["answer"]]
    else:
        new = NEW[task]; pool = ev.rows("fb_pool"); teach = ev.rows("teach"); nid = set(json.load(open(f"/root/data/{DS[task]}_neighbours.json")))
        ex = ev.rows("exam"); base = json.load(open(f"/r/res/base_{DS[task]}_exam_1.json")); neigh = [r for r, o in zip(ex, base["outs"]) if r["id"] in nid and NORM(o) == r["answer"]]
        # neighbours must also be correct for the packed body pre-edit? reported as is (retention relative to base-correct); pre-edit correctness reported by 'empty'
    log("new", new, "pool", len(pool), "teach", len(teach), "neigh", len(neigh)); fb = l4fb.FB(lm, tok, cal, cfgs[0]); results = []
    for c in cfgs:
        fb.cfg = dict(l4fb.DEF, **c); t1 = time.time()
        r = run_draws(ev, fb, task, new, pool, teach, neigh, old_err, fb.cfg, draws, conds, label=c.get("name", "")); r["secs"] = time.time() - t1; results.append(r); conds = [x for x in conds if x != "empty"]
        json.dump(results, open(f"/a/{task}/fb_{tag}.json", "w")); art.commit(); print("RESULT_JSON", json.dumps(r), flush=True)
    return results

def main(task, body, stage, cfg, tag, art):
    os.makedirs(f"/a/{task}", exist_ok=True)
    if stage == "exam": return stage_exam(task, body, cfg, tag, art)
    return stage_fb(task, body, cfg, tag, art, dev=(stage == "fbdev"))
