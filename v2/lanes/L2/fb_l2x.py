"""fb_l2x.py: L2 feedback experiments on the unfolded Ternary-Bonsai-2-27B (bf16). One model load, many jobs.
Closed-form (covariance-protected, Woodbury) edit of EXISTING projection tensors, stored as exact low-rank factors (fp16). No gradient steps, no head/adapter/prompt.
Carriers: 'down:L' (mlp.down_proj), 'o:L' (self_attn.o_proj, full-attention layers only), 'out:L' (linear_attn.out_proj).
usage: python fb_l2x.py JOBS.json OUT.jsonl [--model DIR]
job = {name, data: DATASET, carriers: ["down:62","down:63"], m, lam, f, n, draws, arms: [...], store: "fp16"|"fp32", persist: false, rounds: [...]}
DATASET dir (jsonl, rows {user, answer, id, src, kind}): ann.jsonl teach.jsonl neigh.jsonl cal.jsonl [old.jsonl]; DATASET.json: {new, shuf_labels, ann_filter}
arms: empty | real{n} (all-new-label annotations) | mreal | mshuf | mwrong | unrel   (see ARMS below).
Every arm line is appended to OUT.jsonl as soon as it is done."""
import os, sys, json, random, time, math, collections, argparse
import torch
from transformers import AutoTokenizer, AutoModelForImageTextToText
ap = argparse.ArgumentParser(); ap.add_argument("jobs"); ap.add_argument("out"); ap.add_argument("--model", default=os.environ.get("UNFOLDED", "_unfolded")); a = ap.parse_args()
dev = "cuda"
tok = AutoTokenizer.from_pretrained(a.model); tok.padding_side = "left"
model = AutoModelForImageTextToText.from_pretrained(a.model, dtype=torch.bfloat16, device_map={"": 0}).eval()
base_model = model.model if hasattr(model, "model") else model
LM = getattr(base_model, "language_model", base_model); LAY = LM.layers; HEAD = model.lm_head; NORM = LM.norm
def site_mod(spec):
    k, L = spec.split(":"); lay = LAY[int(L)]
    return {"down": lambda: lay.mlp.down_proj, "o": lambda: lay.self_attn.o_proj, "out": lambda: lay.linear_attn.out_proj}[k]()
def prompt_ids(u): return tok.apply_chat_template([{"role": "user", "content": u}], add_generation_prompt=True, tokenize=True, enable_thinking=False, return_dict=False)
def ids_of(x):
    r = prompt_ids(x); return r["input_ids"] if isinstance(r, dict) else list(r)
@torch.no_grad()
def gen(items, mn=6, bs=48):
    outs = [None] * len(items); order = sorted(range(len(items)), key=lambda i: len(items[i]["ids"]))
    for b in range(0, len(order), bs):
        idx = order[b:b + bs]; L = max(len(items[i]["ids"]) for i in idx)
        ids = torch.full((len(idx), L), tok.pad_token_id or 0, dtype=torch.long); am = torch.zeros((len(idx), L), dtype=torch.long)
        for r, i in enumerate(idx): s = items[i]["ids"]; ids[r, L - len(s):] = torch.tensor(s); am[r, L - len(s):] = 1
        g = model.generate(input_ids=ids.to(dev), attention_mask=am.to(dev), max_new_tokens=mn, do_sample=False, pad_token_id=tok.pad_token_id or 0)
        for r, i in enumerate(idx): outs[i] = tok.decode(g[r, L:], skip_special_tokens=True).strip()
    return outs
def word(o): return (o.split() or [""])[0].strip(".,:;!\"'").lower()
def capture(seqs, spec, budget=6000):
    out = [None] * len(seqs); order = sorted(range(len(seqs)), key=lambda i: len(seqs[i][0])); i = 0; buf = {}
    h1 = site_mod(spec).register_forward_pre_hook(lambda m, ar: buf.__setitem__("k", ar[0]))
    h2 = NORM.register_forward_pre_hook(lambda m, ar: buf.__setitem__("x", ar[0]))
    with torch.no_grad():
        while i < len(order):
            j = i
            while j < len(order) and (j - i + 1) * len(seqs[order[j]][0]) <= budget: j += 1
            j = max(j, i + 1); idx = order[i:j]; Lm = max(len(seqs[k][0]) for k in idx)
            ids = torch.full((len(idx), Lm), tok.pad_token_id or 0, dtype=torch.long); am = torch.zeros((len(idx), Lm), dtype=torch.long)
            for r, k in enumerate(idx): s = seqs[k][0]; ids[r, :len(s)] = torch.tensor(s); am[r, :len(s)] = 1
            h = LM(input_ids=ids.to(dev), attention_mask=am.to(dev)).last_hidden_state
            for r, k in enumerate(idx):
                ps = seqs[k][1]; x = buf["x"][r, ps].float(); out[k] = (h[r, ps].clone(), buf["k"][r, ps].float().clone(), (x.pow(2).mean(-1, keepdim=True) + 1e-6).sqrt())
            i = j
    h1.remove(); h2.remove(); return out
def mkseq(r, label):
    a = tok(label + "<|im_end|>", add_special_tokens=False)["input_ids"]; ids = r["ids"] + a; pos = list(range(len(r["ids"]) - 1, len(ids) - 1)); return ids, pos, a
COV = {}
def cov(spec, cal):
    if spec in COV: return COV[spec]
    mod = site_mod(spec); dff = mod.weight.shape[1]; C = torch.zeros(dff, dff, device=dev); n = 0; buf = {}
    hk = mod.register_forward_pre_hook(lambda m, ar: buf.__setitem__("k", ar[0]))
    seqs = [r["ids"] + tok(r["answer"] + "<|im_end|>", add_special_tokens=False)["input_ids"] for r in cal]; order = sorted(range(len(seqs)), key=lambda i: len(seqs[i])); i = 0
    with torch.no_grad():
        while i < len(order):
            j = i
            while j < len(order) and (j - i + 1) * len(seqs[order[j]]) <= 6000: j += 1
            j = max(j, i + 1); idx = order[i:j]; Lm = max(len(seqs[k]) for k in idx)
            ids = torch.full((len(idx), Lm), tok.pad_token_id or 0, dtype=torch.long); am = torch.zeros((len(idx), Lm), dtype=torch.long)
            for r, k in enumerate(idx): ids[r, :len(seqs[k])] = torch.tensor(seqs[k]); am[r, :len(seqs[k])] = 1
            LM(input_ids=ids.to(dev), attention_mask=am.to(dev)); kk = buf["k"][am.bool().to(dev)].float(); C += kk.T @ kk; n += len(kk); i = j
    hk.remove(); COV[spec] = C / n; return COV[spec]
with torch.no_grad(): GN = NORM(torch.ones(1, NORM.weight.shape[0], device=dev, dtype=torch.bfloat16)).float()[0] * math.sqrt(1 + 1e-6)
def plan(items, labels, spec, M):
    sq = [mkseq(r, lab) for r, lab in zip(items, labels)]; caps = capture([(x, y) for x, y, _ in sq], spec); Rs = []; Ks = []; W = HEAD.weight
    for (h, k, rms), (_, _, g) in zip(caps, sq):
        lg = (h.to(W.dtype) @ W.T).float()
        for j in range(len(g)):
            t = int(g[j]); top = int(lg[j].argmax())
            if top == t: continue
            d = (W[t] - W[top]).float(); gap = float(lg[j, top] - lg[j, t]); c = (gap + M) * float(rms[j]) / float((GN * d).pow(2).sum()); Rs.append(c * GN * d); Ks.append(k[j])
    return (torch.stack(Rs, 1), torch.stack(Ks, 1)) if Rs else None
SAVED = {}
def restore():
    for s, w in SAVED.items(): site_mod(s).weight.data.copy_(w)
    SAVED.clear()
def apply_edit(items, labels, sites, M, LAM, F, store, cal):
    """sequential over sites (re-plan after each so later sites see the already edited model). returns (factors {spec:(U,V)}, bytes)"""
    fac = {}; nbytes = 0
    for idx_s, spec in enumerate(sites):
        p = plan(items, labels, spec, M)
        if p is None: continue
        R, K = p; C = cov(spec, cal)
        for eps in (1e-3, 1e-2, 1e-1, 1.0):
            try: Lc = torch.linalg.cholesky(LAM * C + eps * LAM * C.diagonal().mean() * torch.eye(C.shape[0], device=dev)); break
            except Exception: pass
        B = torch.cholesky_solve(K, Lc); S = torch.eye(K.shape[1], device=dev) + K.T @ B; sc = F / (len(sites) - idx_s)
        Vt = torch.linalg.solve(S, B.T); U = (sc * R); V = Vt.T.contiguous()
        if store == "fp16": U = U.half().float(); V = V.half().float()
        mod = site_mod(spec); P = mod.weight
        if spec not in SAVED: SAVED[spec] = P.data.clone()
        P.data.copy_((P.data.float() + U @ V.T).to(P.dtype)); fac[spec] = (U.cpu(), V.cpu()); nbytes += (U.numel() + V.numel()) * (2 if store == "fp16" else 4)
    return fac, nbytes
# ---------------- data
def rd(f): return [json.loads(l) for l in open(f)] if os.path.exists(f) else []
def load_ds(d):
    S = {n: rd(f"{d}/{n}.jsonl") for n in ("ann", "teach", "neigh", "cal", "old")}; meta = json.load(open(f"{d}/DATASET.json"))
    for v in S.values():
        for r in v: r["ids"] = ids_of(r["user"])
    return S, meta
DS = {}
def getds(d):
    if d not in DS: DS[d] = load_ds(d)
    return DS[d]
def evaluate(tag, teachsets, NEI, newwords):
    res = dict(tag=tag); no = gen(NEI); nw = [word(o) for o in no]
    for name, (items, new) in teachsets.items():
        to = gen(items); res["teach_" + name] = sum(word(o) == new.lower() for o in to) / len(items)
        res["outs_" + name] = collections.Counter(word(o) for o in to).most_common(4)
    neigh = [w == r["answer"].lower() for w, r in zip(nw, NEI)]
    cells = collections.defaultdict(list)
    for ok, r in zip(neigh, NEI): cells[(r["src"], r["answer"])].append(ok)
    res.update(neigh=sum(neigh) / len(neigh), min_cell=min(sum(v) / len(v) for v in cells.values() if len(v) >= 15), new_on_neigh=sum(w in newwords for w in nw) / len(nw), n_neigh=len(NEI))
    return res
def derange(rng, labs, maxfix=2):
    for _ in range(1000):
        p = labs[:]; rng.shuffle(p)
        if sum(x == y for x, y in zip(p, labs)) <= maxfix: return p
    return p
def build_arm(arm, rng, ANN, OLD, new, old_labels):
    n = int(''.join(c for c in arm if c.isdigit()) or 25); base = arm.rstrip("0123456789")
    if base == "real": sel = rng.sample(ANN, min(n, len(ANN))); return sel, [new] * len(sel)
    if base == "unrel": sel = rng.sample(OLD, 25); return sel, [new] * 25
    k = 13; A = rng.sample(ANN, k); O = rng.sample(OLD, 25 - k); items = A + O; labs = [new] * k + [o["answer"] for o in O]
    if base == "mreal": return items, labs
    if base == "mshuf": return items, derange(rng, labs)
    if base == "mwrong": return items, [rng.choice([l for l in old_labels]) for _ in A] + labs[k:]
    raise ValueError(arm)
open(a.out, "a").close(); JOBS = json.load(open(a.jobs))
def emit(r): open(a.out, "a").write(json.dumps(r) + "\n"); print(json.dumps(r)[:400], flush=True)
for job in JOBS:
    t0 = time.time(); (S, meta) = getds(job["data"]); new = meta["new"]; M = job.get("m", 32); LAM = job.get("lam", 1.0); F = job.get("f", 1.0); st = job.get("store", "fp16"); sites = job["carriers"]
    restore(); COV.clear()
    base_neigh = gen(S["neigh"]); NEI = [r for r, o in zip(S["neigh"], base_neigh) if word(o) == r["answer"].lower()][:600]
    base_teach = gen(S["teach"]); TEACH = [r for r, o in zip(S["teach"], base_teach) if word(o) != new.lower()]
    ann_out = gen(S["ann"]); ANN = [r for r, o in zip(S["ann"], ann_out) if word(o) != new.lower()]
    OLD = S["old"]; old_labels = meta.get("shuf_labels", ["Alarm", "Weather", "Music", "Calendar", "Lists", "News", "Transport"])
    teachsets = {new: (TEACH, new)}; newwords = {new.lower()}
    for e in job.get("extra_teach", []):   # (dataset dir) additional new labels to read out (continual rounds)
        S2, m2 = getds(e); b = gen(S2["teach"]); teachsets[m2["new"]] = ([r for r, o in zip(S2["teach"], b) if word(o) != m2["new"].lower()], m2["new"]); newwords.add(m2["new"].lower())
    for s in sites: cov(s, S["cal"])
    print("JOB", job["name"], "teach n", len(TEACH), "neigh", len(NEI), "ann usable", len(ANN), "prep secs", time.time() - t0, flush=True)
    for arm in job["arms"]:
        for d in range(1 if arm == "empty" else job.get("draws", 3)):
            rng = random.Random(7919 * (d + 1) + job.get("seed", 0)); restore(); t1 = time.time()
            if arm == "empty": r = evaluate("empty", teachsets, NEI, newwords); r.update(job=job["name"], arm="empty", draw=d); emit(r); continue
            items, labs = build_arm(arm, rng, ANN, OLD, new, old_labels); fac, nb = apply_edit(items, labs, sites, M, LAM, F, st, S["cal"])
            r = evaluate(f"{arm}-d{d}", teachsets, NEI, newwords); r.update(job=job["name"], arm=arm, draw=d, bytes=nb, ranks={s: u.shape[1] for s, (u, v) in fac.items()}, m=M, lam=LAM, carriers=sites, secs=time.time() - t1)
            if job.get("save") and d == 0 and arm == job.get("save_arm", arm):
                torch.save({f"model.language_model.layers.{s.split(':')[1]}.{ {'down': 'mlp.down_proj', 'o': 'self_attn.o_proj', 'out': 'linear_attn.out_proj'}[s.split(':')[0]] }.weight": (u.half(), v.half()) if st == "fp16" else (u, v) for s, (u, v) in fac.items()}, job["save"]); r["saved"] = job["save"]
            emit(r)
    # continual: round 1 = job arm 'real', round 2 = second dataset, edits stacked on one model
    if job.get("continual"):
        c = job["continual"]; S2, m2 = getds(c["data2"]); new2 = m2["new"]; b2 = gen(S2["teach"])
        for d in range(c.get("draws", 3)):
            rng = random.Random(7919 * (d + 1) + 5); restore(); COV.clear()
            for s in sites: cov(s, S["cal"])
            i1, l1 = build_arm("real25", rng, ANN, OLD, new, old_labels); fac1, nb1 = apply_edit(i1, l1, sites, M, LAM, F, st, S["cal"])
            r1 = evaluate(f"cont-r1-d{d}", teachsets, NEI, newwords); r1.update(job=job["name"], arm="cont_round1", draw=d, bytes=nb1); emit(r1)
            ANN2 = [r for r, o in zip(S2["ann"], gen(S2["ann"])) if word(o) != new2.lower()]
            COV.clear(); [cov(s, S["cal"]) for s in sites]    # covariance of the model that already carries round 1
            i2 = rng.sample(ANN2, 25); l2 = [new2] * 25
            if c.get("replay"): i2 = i2[:13] + rng.sample(ANN, 12); l2 = [new2] * 13 + [new] * 12
            fac2, nb2 = apply_edit(i2, l2, sites, M, LAM, F, st, S["cal"])
            r2 = evaluate(f"cont-r2-d{d}", teachsets, NEI, newwords); r2.update(job=job["name"], arm="cont_round2" + ("_replay" if c.get("replay") else ""), draw=d, bytes=nb1 + nb2); emit(r2)
            if c.get("save") and d == 0:
                nm = lambda s: f"model.language_model.layers.{s.split(':')[1]}.{ {'down': 'mlp.down_proj', 'o': 'self_attn.o_proj', 'out': 'linear_attn.out_proj'}[s.split(':')[0]] }.weight"
                torch.save({"round1": {nm(s): (u.half(), v.half()) for s, (u, v) in fac1.items()}, "round2": {nm(s): (u.half(), v.half()) for s, (u, v) in fac2.items()}}, c["save"])
    restore(); print("JOB_DONE", job["name"], time.time() - t0, flush=True)
print("ALL_DONE")
